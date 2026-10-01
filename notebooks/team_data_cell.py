# ======= ДАНІ: командний датасет Anti-UAV410 SKY + синтетичне сонце =======
DRIVE_URL = 'https://drive.google.com/file/d/1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR/view'   # AntiUAV410.zip (сирий архів)
PKG_URL = ''   # не потрібен: пакет команди (маніфест відібраних кадрів + builder) вбудовано в клітинку вище
TRAIN_EVERY = 1   # маніфест у клітинці вище вже проріджений (__THIN__), тут більше не проріджуємо
VAL_EVERY = 1     # val у маніфесті вже кожен 4-й кадр. Test тут не збираємо: фінальний тест робимо на 2080 SUPER з повним пакетом
SPLITS = __SPLITS__
SUN_FRAC = 0.10                 # до train додаємо копії 10% кадрів із синтетичним сонцем (ціль лишається, сонце за 40+ px від неї)

import os, glob, re, subprocess, zipfile, shutil, random, yaml
import numpy as np, pandas as pd, cv2
DL = '/tmp/dl'; os.makedirs(DL, exist_ok=True)

def fetch(url, out):
    # Великі файли Google Drive віддає через сторінку "не вдалося перевірити на віруси".
    # Качаємо напряму з drive.usercontent.google.com з confirm=t, а якщо прийшла сторінка, беремо з неї форму і пробуємо ще раз.
    import requests
    if os.path.exists(out) and os.path.getsize(out) > 1 << 20: print('вже є', out); return
    fid = (re.search(r'/d/([\w-]{20,})', url) or re.search(r'id=([\w-]{20,})', url)).group(1)
    s, u, params = requests.Session(), 'https://drive.usercontent.google.com/download', {'id': fid, 'export': 'download', 'confirm': 't'}
    for _ in range(3):
        r = s.get(u, params=params, stream=True, timeout=120)
        if 'text/html' not in r.headers.get('Content-Type', ''):
            total, done, mark = int(r.headers.get('Content-Length', 0)), 0, 0
            with open(out, 'wb') as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk); done += len(chunk)
                    if done - mark > 1 << 30: mark = done; print(f'  {done / 2**30:.1f} / {total / 2**30:.1f} ГБ', flush=True)
            print(f'скачано {out}: {done / 2**30:.2f} ГБ'); return
        html = r.text
        form = re.search(r'<form[^>]+action="([^"]+)"', html)
        inputs = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html))
        if not (form and inputs): break
        u, params = form.group(1).replace('&amp;', '&'), inputs
    t = re.findall(r'<title>(.*?)</title>', html, re.S)
    raise SystemExit('Google Drive не віддав файл: ' + (t[0].strip() if t else '?') + '. Перевірте доступ "Усі, хто має посилання".')

# 1. Сирий архів і пакет команди
RAW = f'{DL}/AntiUAV410.zip'; fetch(DRIVE_URL, RAW)
assert zipfile.is_zipfile(RAW), 'DRIVE_URL має вести на zip-архів Anti-UAV410'
if not os.path.exists(f'{DL}/team_package/build_antiuav_yolo.py') and PKG_URL.startswith('http'):
    fetch(PKG_URL, f'{DL}/pkg.zip'); zipfile.ZipFile(f'{DL}/pkg.zip').extractall(DL)
PKG = os.path.dirname(glob.glob(f'{DL}/**/build_antiuav_yolo.py', recursive=True)[0])
subprocess.run('pip install -q "pandas>=2.2" "Pillow>=10" "PyYAML>=6" tqdm', shell=True)

# 2. Проріджуємо маніфест: train кожен TRAIN_EVERY-й позитив у серії, val кожен VAL_EVERY-й, негативи і test усі
m = pd.read_csv(f'{PKG}/selected_frames.csv.gz', dtype={'sequence': str, 'image_name': str, 'sample_role': str, 'assigned_split': str})
m = m[m.assigned_split.isin(SPLITS)].sort_values(['assigned_split', 'sequence', 'frame_number'])
pos = m.sample_role == 'positive'
rank = m[pos].groupby(['assigned_split', 'sequence']).cumcount()
step = m.loc[pos, 'assigned_split'].map({'train': TRAIN_EVERY, 'val': VAL_EVERY, 'test': 1})
keep = pd.Series(True, index=m.index); keep[pos] = (rank % step == 0)
m = m[keep]
MAN = f'{DL}/manifest_used.csv'; m.to_csv(MAN, index=False)
print('кадрів у збірці:', m.groupby(['assigned_split', 'sample_role']).size().to_dict())

# 3. Збірка командним builder: перевірка шляхів, потім повна збірка (кожен JPEG декодується і перевіряється)
SRC = '/tmp/sky_yolo'
if not os.path.exists(f'{SRC}/data.yaml'):
    shutil.rmtree('/tmp/sky_check', ignore_errors=True); shutil.rmtree(SRC, ignore_errors=True)
    r = subprocess.run(f'python {PKG}/build_antiuav_yolo.py --source {RAW} --manifest {MAN} --output /tmp/sky_check --check-only', shell=True)
    if r.returncode != 0:
        if os.path.exists('/tmp/sky_check/preflight_errors.csv'): print(open('/tmp/sky_check/preflight_errors.csv').read()[:3000])
        raise SystemExit('Перевірка архіву не пройшла: шляхи в архіві не збігаються з маніфестом. Можливо, на Drive інший архів, ніж у того, хто робив пакет.')
    r = subprocess.run(f'python {PKG}/build_antiuav_yolo.py --source {RAW} --manifest {MAN} --output {SRC}', shell=True)
    errs = pd.read_csv(f'{SRC}/build_errors.csv')
    print('помилок збірки:', len(errs))
    if len(errs) > 0.01 * len(m): raise SystemExit('Забагато помилок збірки, див. /tmp/sky_yolo/build_errors.csv')

# 4. Синтетичне сонце: копії частини кадрів train із пересвіченим диском і ореолом
def add_sun(img, boxes, rng):
    h, w = img.shape
    r = rng.uniform(8, 30); sig = rng.uniform(1.0, 3.0) * r
    for _ in range(200):
        if boxes and rng.random() < 0.5:                       # половина сонць поруч із ціллю (40-120 px)
            bx, by = boxes[rng.integers(len(boxes))]; ang = rng.uniform(0, 2 * np.pi); d = rng.uniform(40, 120) + r
            cx, cy = bx + d * np.cos(ang), by + d * np.sin(ang)
        else:
            cx, cy = rng.uniform(r, w - r), rng.uniform(r, h * 0.6)
        if not (r <= cx < w - r and r <= cy < h - r): continue
        if boxes and min(np.hypot(bx - cx, by - cy) for bx, by in boxes) < 40 + r: continue
        break
    yy, xx = np.mgrid[:h, :w]; d = np.hypot(xx - cx, yy - cy)
    halo = np.where(d <= r, 1.0, np.exp(-((d - r) ** 2) / (2 * sig ** 2)))
    out = img.astype(np.float32) + (255 - img.astype(np.float32)) * halo * rng.uniform(0.6, 1.0)
    out[d <= r] = 255
    return np.clip(out, 0, 255).astype(np.uint8)

if SUN_FRAC > 0 and not glob.glob(f'{SRC}/images/train/sun_*'):
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(f'{SRC}/images/train/*.jpg')); random.Random(0).shuffle(files)
    for p in files[:int(len(files) * SUN_FRAC)]:
        n = os.path.basename(p)[:-4]; img = cv2.imread(p, 0); H, W = img.shape
        lab = open(f'{SRC}/labels/train/{n}.txt').read()
        centers = [(float(l.split()[1]) * W, float(l.split()[2]) * H) for l in lab.splitlines() if l.strip()]
        cv2.imwrite(f'{SRC}/images/train/sun_{n}.jpg', add_sun(img, centers, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        open(f'{SRC}/labels/train/sun_{n}.txt', 'w').write(lab)

d = yaml.safe_load(open(f'{SRC}/data.yaml')); d['path'] = SRC
yaml.safe_dump(d, open('/kaggle/working/data.yaml', 'w'))
for s in SPLITS:
    labs = glob.glob(f'{SRC}/labels/{s}/*.txt')
    neg = sum(os.path.getsize(f) == 0 for f in labs); sun = len(glob.glob(f'{SRC}/images/{s}/sun_*'))
    print(f'{s}: {len(labs)} кадрів, негативів {neg} ({neg / max(len(labs), 1):.1%}), із сонцем {sun}')
