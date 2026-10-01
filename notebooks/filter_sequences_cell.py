# ===== ФІЛЬТР СЕРІЙ: лишаємо серії, де ціль на небі і розміром 5-50 px =====
SKY_GRAD = 25      # макс. середній градієнт фону навколо цілі: небо гладке, будівлі й дерева ні
MIN_SKY = 0.6      # серію лишаємо, якщо ціль на небі хоча б у 60% її кадрів
MIN_SIZE = 0.5     # і якщо ціль 5-50 px хоча б у 50% кадрів
import os, glob, csv, collections, cv2, numpy as np

def bg_grad(img, x, y, w, h, ring=12):
    H, W = img.shape
    X0, Y0, X1, Y1 = max(x - ring - 2, 0), max(y - ring - 2, 0), min(x + w + ring + 2, W), min(y + h + ring + 2, H)
    P = img[Y0:Y1, X0:X1].astype(np.float32)
    g = np.hypot(cv2.Sobel(P, cv2.CV_32F, 1, 0), cv2.Sobel(P, cv2.CV_32F, 0, 1))
    inner = np.zeros(P.shape, bool); inner[max(y - 2 - Y0, 0):max(y + h + 2 - Y0, 0), max(x - 2 - X0, 0):max(x + w + 2 - X0, 0)] = True
    return float(g[~inner].mean()) if (~inner).any() else 0.0

stats = collections.defaultdict(lambda: [0, 0, 0])          # (split, серія) -> [кадрів, 5-50 px, на небі]
for split in ['train', 'val', 'test']:
    for lf in glob.glob(f'{SRC}/labels/{split}/*.txt'):
        n = os.path.basename(lf)[:-4]
        line = open(lf).readline().split()
        if n.startswith('neg_') or not line: continue
        _, cx, cy, w, h = map(float, line)
        img = cv2.imread(f'{SRC}/images/{split}/{n}.jpg', 0); H, W = img.shape
        bw, bh = w * W, h * H
        s = stats[(split, n.rsplit('_', 1)[0])]
        s[0] += 1; s[1] += 5 <= max(bw, bh) <= 50
        s[2] += bg_grad(img, int(cx * W - bw / 2), int(cy * H - bh / 2), max(int(round(bw)), 1), max(int(round(bh)), 1)) <= SKY_GRAD

def dropped(size_only=False):
    return {k for k, (n, ok, sky) in stats.items() if ok / n < MIN_SIZE or (not size_only and sky / n < MIN_SKY)}
drop = dropped()
kept_train = sum(v[0] for k, v in stats.items() if k[0] == 'train' and k not in drop)
all_train = sum(v[0] for k, v in stats.items() if k[0] == 'train')
if kept_train < 0.3 * all_train:                              # страховка: якщо поріг неба зрізав майже все, фільтруємо лише за розміром
    print(f'УВАГА: фільтр неба лишив би {kept_train} з {all_train} кадрів train, фільтрую лише за розміром цілі')
    drop = dropped(size_only=True)

with open('/kaggle/working/sequences.csv', 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['split', 'sequence', 'frames', 'share_5_50', 'share_sky', 'kept'])
    for (sp, seq), (n, ok, sky) in sorted(stats.items()):
        wr.writerow([sp, seq, n, round(ok / n, 2), round(sky / n, 2), (sp, seq) not in drop])
for sp, seq in drop:
    for p in glob.glob(f'{SRC}/images/{sp}/{seq}_*') + glob.glob(f'{SRC}/labels/{sp}/{seq}_*'):
        os.remove(p)
for sp in ['train', 'val', 'test']:
    ks = [k for k in stats if k[0] == sp]
    print(f'{sp}: серій лишили {sum(k not in drop for k in ks)} з {len(ks)}, кадрів з ціллю {sum(stats[k][0] for k in ks if k not in drop)}, '
          f'негативів {len(glob.glob(f"{SRC}/images/{sp}/neg_*"))}')
