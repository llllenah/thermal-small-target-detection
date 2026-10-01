"""Збирає два Kaggle-ноутбуки: yolo_kaggle.ipynb і dnanet_kaggle.ipynb."""
import json, os
D = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(D)
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def wf(path, src):
    """Файл пишемо звичайним Python, без %%writefile: працює, навіть якщо клітинки склеїли в одну."""
    t = open(src).read(); assert "'''" not in t
    return code(f"import os; os.makedirs(os.path.dirname({path!r}), exist_ok=True)\nopen({path!r}, 'w').write(r'''" + t + "'''); print('записано', " + repr(path) + ")")
import base64
_TP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_package")
def PKG_CELL(man):
    return ("# ======= ПАКЕТ КОМАНДИ AntiUAV410_SKY: builder і маніфест відібраних кадрів (вже проріджений під цей ноутбук) =======\n"
            "import os, base64\nos.makedirs('/tmp/dl/team_package', exist_ok=True)\n"
            "open('/tmp/dl/team_package/build_antiuav_yolo.py', 'w').write(r\'\'\'" + open(f"{_TP}/build_antiuav_yolo.py").read() + "\'\'\')\n"
            "open('/tmp/dl/team_package/selected_frames.csv.gz', 'wb').write(base64.b64decode('"
            + base64.b64encode(open(f"{_TP}/{man}", "rb").read()).decode() + "'))\n"
            "print('пакет команди записано')")
TEAM = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_data_cell.py")).read()
FILT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "filter_sequences_cell.py")).read()
META = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}

FIND = r"""# ======= ДАНІ: посилання на Google Drive =======
DRIVE_URL = 'ВСТАВИТИ_ПОСИЛАННЯ'   # zip з Anti-UAV410 (train/val/test) АБО вже готовий YOLO-датасет; доступ "Усі, хто має посилання"
NEG_URL = ''                         # необов'язково: посилання на negatives_selected.zip
STRIDE, NEG_FRAC = 20, 0.1           # кожен 20-й кадр із ціллю, 10% негативів

import os, glob, subprocess, zipfile, tarfile, yaml, shutil
subprocess.run('pip install -q gdown', shell=True)
DL = '/tmp/dl'; os.makedirs(DL, exist_ok=True)

def fetch(url, dst):
    # Великі файли Google Drive віддає через сторінку "не вдалося перевірити на віруси". Качаємо напряму з
    # drive.usercontent.google.com з confirm=t, а якщо прийшла сторінка, беремо з неї форму підтвердження і пробуємо ще раз.
    import re, requests
    os.makedirs(dst, exist_ok=True)
    if '/folders/' in url:
        subprocess.run(['gdown', '--folder', '-O', dst, url], check=True); return
    fid = (re.search(r'/d/([\w-]{20,})', url) or re.search(r'id=([\w-]{20,})', url)).group(1)
    out, s = dst + '/download', requests.Session()
    u, params = 'https://drive.usercontent.google.com/download', {'id': fid, 'export': 'download', 'confirm': 't'}
    for _ in range(3):
        r = s.get(u, params=params, stream=True, timeout=120)
        if 'text/html' not in r.headers.get('Content-Type', ''):
            total, done, mark = int(r.headers.get('Content-Length', 0)), 0, 0
            with open(out, 'wb') as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk); done += len(chunk)
                    if done - mark > 1 << 30: mark = done; print(f'  {done / 2**30:.1f} / {total / 2**30:.1f} ГБ', flush=True)
            print(f'скачано {done / 2**30:.2f} ГБ'); return
        html = r.text
        form = re.search(r'<form[^>]+action="([^"]+)"', html)
        inputs = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html))
        if not (form and inputs): break
        u, params = form.group(1).replace('&amp;', '&'), inputs; print('Drive попросив підтвердження, пробую ще раз')
    open(out, 'w').write(html)                          # check() нижче покаже, що саме відповів Drive

def check(folder):
    # gdown інколи зберігає файл без розширення: визначаємо тип за першими байтами.
    # Якщо замість архіву скачалась HTML-сторінка, це відмова Google Drive: немає доступу або вичерпана квота.
    for f in glob.glob(folder + '/*'):
        if os.path.isdir(f) or os.path.splitext(f)[1]: continue
        head = open(f, 'rb').read(600)
        ext = ('.zip' if head[:2] == b'PK' else '.tar.gz' if head[:2] == b'\x1f\x8b' else '.7z' if head[:4] == b"7z\xbc\xaf"
               else '.rar' if head[:4] == b'Rar!' else '.tar' if head[257:262] == b'ustar' else None)
        if ext:
            os.rename(f, f + ext); print('тип файлу:', ext)
        elif b'<html' in head.lower() or b'<!doctype' in head.lower():
            import re; t = re.findall(rb'<title>(.*?)</title>', open(f, 'rb').read(), re.S)
            raise SystemExit('Google Drive повернув сторінку замість файлу: ' + (t[0].decode(errors="ignore").strip() if t else '?') +
                             '. Перевірте: 1) Поділитися -> "Усі, хто має посилання"; 2) це файл-архів, а не папка чи Google-документ; '
                             '3) якщо "Quota exceeded", скопіюйте файл на інший Drive або додайте архів як Kaggle Input.')
        else:
            raise SystemExit(f'Невідомий тип файлу {f}, перші байти: {head[:16]}')

def unpack(root):
    for _ in range(3):                                   # архів в архіві
        arcs = [f for f in glob.glob(root + '/**/*', recursive=True) if f.endswith(('.zip', '.tar', '.tar.gz', '.tgz', '.7z', '.rar'))]
        if not arcs: break
        for f in arcs:
            print('розпаковую', f, round(os.path.getsize(f) / 2**30, 2), 'ГБ')
            if f.endswith('.zip'): zipfile.ZipFile(f).extractall(os.path.dirname(f))
            elif f.endswith(('.7z', '.rar')): subprocess.run(f'7z x -y -o"{os.path.dirname(f)}" "{f}" > /dev/null || (pip install -q py7zr && python -m py7zr x "{f}" "{os.path.dirname(f)}")', shell=True, check=True)
            else: tarfile.open(f).extractall(os.path.dirname(f))
            os.remove(f)

have = glob.glob(DL + '/data/**/data.yaml', recursive=True) + glob.glob(DL + '/data/**/IR_label.json', recursive=True)
if DRIVE_URL.startswith('http') and not have:        # повторний запуск після невдалої спроби качає заново
    shutil.rmtree(DL + '/data', ignore_errors=True); fetch(DRIVE_URL, DL + '/data')
if NEG_URL.startswith('http') and not os.path.exists(DL + '/neg'):
    fetch(NEG_URL, DL + '/neg')
os.makedirs(DL + '/data', exist_ok=True)
for f in glob.glob('/kaggle/input/**/*.zip', recursive=True):   # архіви, додані як Kaggle Input
    shutil.copy(f, DL + '/data/')
check(DL + '/data'); check(DL + '/neg')
unpack(DL)
!df -h /tmp | tail -1

ready = sorted(glob.glob(DL + '/**/data.yaml', recursive=True) + glob.glob('/kaggle/input/**/data.yaml', recursive=True))
raw = sorted(glob.glob(DL + '/**/IR_label.json', recursive=True) + glob.glob('/kaggle/input/**/IR_label.json', recursive=True))
if ready:
    SRC = os.path.dirname(ready[0]); print('готовий YOLO-датасет:', SRC)
elif raw:
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(raw[0])))   # .../<root>/<split>/<серія>/IR_label.json
    NEG = next((d for d in glob.glob(DL + '/**/negatives', recursive=True) + glob.glob('/kaggle/input/**/negatives', recursive=True)
                if os.path.isdir(d + '/train')), None)
    print('сирий Anti-UAV410:', ROOT, '| негативи:', NEG or 'немає, вчимо без них')
    SRC = '/tmp/auav_yolo'; NEGARG = f'--neg {NEG}' if NEG else ''
    if not os.path.exists(SRC + '/data.yaml'):
        !python /kaggle/working/prepare_yolo.py --root {ROOT} --out {SRC} --stride {STRIDE} --neg_frac {NEG_FRAC} {NEGARG}
else:
    raise SystemExit('Не знайшов ні data.yaml, ні IR_label.json. Перевірте посилання і доступ "Усі, хто має посилання". Що скачалось: '
                     + str(sorted(glob.glob(DL + '/data/**', recursive=True))[:30]))
d = yaml.safe_load(open(SRC + '/data.yaml')); d['path'] = SRC
yaml.safe_dump(d, open('/kaggle/working/data.yaml', 'w'))
for s in ['train', 'val', 'test']:
    n = len(glob.glob(f'{SRC}/images/{s}/*')); neg = len(glob.glob(f'{SRC}/images/{s}/neg_*'))
    print(f'{s}: {n} кадрів, з них негативів {neg}')
"""

yolo = [
    md("# YOLO для кейсу 4.5: навчання на Kaggle\n\n"
       "**Перед запуском:** праворуч Settings → Accelerator **GPU T4 x2**, Internet **On**. "
       "У клітинці ДАНІ вже стоять посилання: сирий Anti-UAV410 і пакет команди AntiUAV410_SKY (маніфест відібраних кадрів неба). Додається синтетичне сонце.\n\n"
       "**Запуск:** змінити `MODEL` у клітинці нижче → **Save Version → Save & Run All**. Ноутбук працює у фоні, "
       "вкладку можна закрити. Результат у вкладці Output: `best.pt`, `best.onnx`, `metrics.json`, `results.csv`.\n\n"
       "Хто що запускає: див. README_TEAM.md."),
    code("# ======= ПАРАМЕТРИ: міняти лише тут =======\n"
         "MODEL = 'yolo26m-p2'   # yolo26m | yolo26m-p2 | yolo26l-p2 | yolo26x-p2\n"
         "EPOCHS = 50            # 12 для швидкої перевірки\n"
         "BATCH = 16             # на 2 GPU разом; для yolo26x-p2 ставити 8\n"
         "HOURS = 10.5           # страховка від ліміту Kaggle 12 год: навчання саме зупиниться і збереже best.pt\n"
         "GPUS = [0, 1]          # якщо впаде з помилкою DDP, поставити [0]\n"),
    wf("/kaggle/working/prepare_yolo.py", f"{R}/yolo/prepare_yolo.py"),
    wf("/kaggle/working/run_yolo.py", f"{R}/yolo/run_yolo.py"),
    code(PKG_CELL("man_yolo.csv.gz")),
    code("!nvidia-smi -L\n!pip install -q ultralytics\n%cd /kaggle/working\n" + TEAM.replace("__THIN__", "train: кожен 3-й кадр із ціллю, val: кожен 4-й").replace("__SPLITS__", "['train', 'val']")),
    code("# Готовий датасет зберігаємо в Output: інші акаунти можуть додати його як Input і не качати з Drive ще раз\n"
         "!cd {os.path.dirname(SRC)} && zip -qr /kaggle/working/auav_yolo.zip {os.path.basename(SRC)} && ls -la /kaggle/working/auav_yolo.zip"),
    md("## Навчання\nP2-голова не має готових ваг: модель будується з yaml, а старт з ваг COCO того ж розміру. "
       "Аугментації під тепловізор: без зміни кольору, помірний масштаб, щоб дрібні цілі не зникали."),
    code("import sys; sys.path.insert(0, '/kaggle/working')\n%cd /kaggle/working\n"
         "from run_yolo import build, AUG\n"
         "m, pre = build(MODEL)\n"
         "m.train(data='/kaggle/working/data.yaml', pretrained=pre, epochs=EPOCHS, time=HOURS, imgsz=640, batch=BATCH, device=GPUS,\n"
         "        project='/kaggle/working/runs', name=MODEL, exist_ok=True, patience=15, cos_lr=True, workers=4, save_period=5, **AUG)"),
    md("## Оцінка за ТЗ на val\nПоріг впевненості підбирається так, щоб FP на кадр ≤ 0.7. Recall лише для цілей 5-50 px, "
       "влучання = центр рамки в рамці цілі ±3 px. Швидкість тут на T4 у PyTorch; фінальний FPS міряємо на 2080 SUPER через TensorRT."),
    code("import glob, shutil, json, subprocess\n"
         "best = sorted(glob.glob(f'/kaggle/working/runs/**/{MODEL}/weights/best.pt', recursive=True))[-1]; print(best)\n"
         "shutil.copy(best, '/kaggle/working/best.pt')\n"
         "shutil.copy(os.path.join(os.path.dirname(os.path.dirname(best)), 'results.csv'), '/kaggle/working/results.csv')\n"
         "!python run_yolo.py eval --weights /kaggle/working/best.pt --val {SRC}:val --test {SRC}:val --json /kaggle/working/metrics.json | tail -30\n"
         "from ultralytics import YOLO\n"
         "YOLO('/kaggle/working/best.pt').export(format='onnx', imgsz=[512, 640], simplify=True)\n"
         "r = json.load(open('/kaggle/working/metrics.json'))\n"
         "print(MODEL, 'conf', r['conf'], 'topk', r['topk'], 'val', r['val'], 'T4 PyTorch', r['speed'])"),
]

dna = [
    md("# DNA-Net для кейсу 4.5: навчання на Kaggle, на тих самих даних, що й YOLO\n\n"
       "**Перед запуском:** Settings → Accelerator **GPU T4 x2** (використовується 1 GPU), Internet **On**. "
       "У клітинці ДАНІ вже стоять посилання: сирий Anti-UAV410 і пакет команди AntiUAV410_SKY (маніфест відібраних кадрів неба). Додається синтетичне сонце.\n\n"
       "**Запуск:** Save Version → Save & Run All. Результат у Output: `log/…/DNANet_*.pth.tar`, `metrics.json`.\n\n"
       "Маски DNA-Net робляться з тих самих рамок YOLO: пікселі в рамці, що відрізняються від фону (Otsu), інакше еліпс."),
    code("# ======= ПАРАМЕТРИ =======\n"
         "EPOCHS = 25        # DNA-Net важка: ~20-30 хв на епоху на T4 при ~20 тис. кадрів. 25 епох укладаються в ліміт\n"
         "BATCH = 16\nPATCH = 256\n"
         "HOURS = 10.5       # страховка від ліміту 12 год: навчання зупиниться, лишиться останній чекпойнт\n"
         "VAL_EVERY = 4      # val ще раз проріджуємо (разом з ДАНИМИ це кожен 16-й кадр), щоб тест під час навчання не з'їдав час\n"),
    code("!nvidia-smi -L\n%cd /kaggle/working\n!rm -rf BasicIRSTD && git clone -q https://github.com/XinyiYing/BasicIRSTD.git"),
    wf("/kaggle/working/BasicIRSTD/patch_basicirstd.py", f"{R}/dnanet/patch_basicirstd.py"),
    wf("/kaggle/working/BasicIRSTD/eval_thermal.py", f"{R}/dnanet/eval_thermal.py"),
    wf("/kaggle/working/prepare_yolo.py", f"{R}/yolo/prepare_yolo.py"),
    code(PKG_CELL("man_dna.csv.gz")),
    code("%cd /kaggle/working\n" + TEAM.replace("__THIN__", "train: кожен 6-й кадр із ціллю, val: кожен 4-й").replace("__SPLITS__", "['train', 'val']")),
    code("%cd /kaggle/working/BasicIRSTD\n!python patch_basicirstd.py"),
    md("## Датасет у форматі BasicIRSTD\n`datasets/AUAV/images/*.png`, `masks/*.png`, `img_idx/train_AUAV.txt`, `test_AUAV.txt` (= val)."),
    code("import sys, cv2, numpy as np\nsys.path.insert(0, '/kaggle/working')\nfrom prepare_yolo import box_to_mask\n"
         "OUT = '/tmp/dnanet_ds/AUAV'   # не в /kaggle/working, щоб тисячі png не йшли в Output\n"
         "os.makedirs(f'{OUT}/images', exist_ok=True); os.makedirs(f'{OUT}/masks', exist_ok=True); os.makedirs(f'{OUT}/img_idx', exist_ok=True)\n"
         "lists = {}\n"
         "for split in ['train', 'val']:\n"
         "    files = sorted(glob.glob(f'{SRC}/images/{split}/*'))\n"
         "    if split == 'val': files = files[::VAL_EVERY]\n"
         "    names = []\n"
         "    for p in files:\n"
         "        n = os.path.basename(p).rsplit('.', 1)[0]; img = cv2.imread(p, 0); H, W = img.shape\n"
         "        boxes = []\n"
         "        lf = f'{SRC}/labels/{split}/{n}.txt'\n"
         "        if os.path.exists(lf):\n"
         "            for l in open(lf).read().split('\\n'):\n"
         "                if l.strip():\n"
         "                    _, cx, cy, w, h = map(float, l.split()); boxes.append((cx * W - w * W / 2, cy * H - h * H / 2, w * W, h * H))\n"
         "        cv2.imwrite(f'{OUT}/images/{n}.png', img); cv2.imwrite(f'{OUT}/masks/{n}.png', box_to_mask(img, boxes)); names.append(n)\n"
         "    lists[split] = names\n"
         "open(f'{OUT}/img_idx/train_AUAV.txt', 'w').write('\\n'.join(lists['train']))\n"
         "open(f'{OUT}/img_idx/test_AUAV.txt', 'w').write('\\n'.join(lists['val']))\n"
         "sample = np.stack([cv2.imread(f'{OUT}/images/{n}.png', 0) for n in lists['train'][::max(1, len(lists['train']) // 500)]]).astype(np.float32)\n"
         "MEAN, STD = float(sample.mean()), float(sample.std()); print('train', len(lists['train']), 'val', len(lists['val']), 'mean', MEAN, 'std', STD)"),
    md("## Навчання\nЧекпойнт кожні 5 епох у `/kaggle/working/log`. Якщо час вийде раніше, лишиться останній збережений."),
    code("import subprocess\n"
         "cmd = (f'CUDA_VISIBLE_DEVICES=0 python train.py --model_names DNANet --dataset_names AUAV --batchSize {BATCH} --patchSize {PATCH} --nEpochs {EPOCHS} '\n"
         "       f'--dataset_dir /tmp/dnanet_ds --save /kaggle/working/log --threads 4 --save_every 5 --img_norm_cfg_mean {MEAN} --img_norm_cfg_std {STD}')\n"
         "import os, signal\n"
         "p = subprocess.Popen(cmd, shell=True, start_new_session=True)   # окрема група процесів, щоб при таймауті вбити і сам train.py\n"
         "try:\n"
         "    p.wait(timeout=HOURS * 3600)\n"
         "except subprocess.TimeoutExpired:\n"
         "    os.killpg(p.pid, signal.SIGKILL); p.wait()\n"
         "    print('час вийшов, навчання зупинено, беремо останній чекпойнт')"),
    md("## Оцінка за ТЗ на val\nТі самі правила, що й для YOLO: Recall цілей 5-50 px, FP на кадр, поріг з перебору."),
    code("import re\n"
         "ck = sorted(glob.glob('/kaggle/working/log/AUAV/DNANet_*.pth.tar'), key=lambda p: int(re.findall(r'_(\\d+)\\.pth', p)[0]))[-1]; print(ck)\n"
         "!CUDA_VISIBLE_DEVICES=0 python eval_thermal.py --model DNANet --ckpt {ck} --dataset_dir /tmp/dnanet_ds --subsets AUAV --mean {MEAN} --std {STD} --sun --sweep --json /kaggle/working/metrics_sweep.json | tail -20\n"
         "!CUDA_VISIBLE_DEVICES=0 python eval_thermal.py --model DNANet --ckpt {ck} --dataset_dir /tmp/dnanet_ds --subsets AUAV --mean {MEAN} --std {STD} --sun --topk 3 --fps --json /kaggle/working/metrics.json | tail -20\n"
         "json.dump(dict(mean=MEAN, std=STD, ckpt=ck), open('/kaggle/working/norm.json', 'w'))"),
]
for name, cells in [("yolo_kaggle", yolo), ("dnanet_kaggle", dna)]:
    json.dump({"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 5}, open(f"{D}/{name}.ipynb", "w"), ensure_ascii=False, indent=1)
print("ok")
