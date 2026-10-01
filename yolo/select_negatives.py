"""Відбір негативів (кадрів без цілі) з Anti-UAV410.

1. Групуємо кадри за серією (split_серія_номер.jpg), сусідні майже однакові кадри викидаємо:
   лишаємо кадр, якщо він помітно відрізняється від попереднього залишеного або пройшло >= gap кадрів.
2. Не більше cap кадрів на серію, щоб одна довга серія не забила весь набір.
3. Проганяємо детектор skydet: кадри, де він щось бачить, це або справжня ціль без розмітки
   (треба глянути очима), або hard negative (найцінніші негативи). Для них робимо превʼю з рамками.

  python select_negatives.py --src candidates_no_target_review --out negatives --skydet ../skydet
"""
import os, re, glob, json, argparse, shutil, sys
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--skydet", required=True)
ap.add_argument("--diff", type=float, default=3.0, help="мін. середня різниця яскравості з попереднім залишеним кадром")
ap.add_argument("--gap", type=int, default=50); ap.add_argument("--cap", type=int, default=60)
a = ap.parse_args()
sys.path.insert(0, a.skydet)
from detector import SkyDetector, load_cfg

seqs = {}
for p in sorted(glob.glob(f"{a.src}/*.jpg")):
    m = re.match(r"(train|val|test)_(.+)_(\d+)\.jpg$", os.path.basename(p))
    seqs.setdefault((m[1], m[2]), []).append((int(m[3]), p))

kept = []
for (split, seq), fr in seqs.items():
    last, last_i, k = None, -10 ** 9, []
    for i, p in sorted(fr):
        th = cv2.resize(cv2.imread(p, 0), (80, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        if last is None or np.abs(th - last).mean() >= a.diff or i - last_i >= a.gap:
            k.append((i, p)); last, last_i = th, i
    if len(k) > a.cap:                                    # рівномірно по серії
        k = [k[j] for j in np.linspace(0, len(k) - 1, a.cap).round().astype(int)]
    kept += [(split, seq, i, p) for i, p in k]
print("кадрів", sum(len(v) for v in seqs.values()), "серій", len(seqs), "лишили", len(kept), flush=True)

cfg = load_cfg(f"{a.skydet}/config.json"); cfg["model"] = f"{a.skydet}/{cfg['model']}"
det = SkyDetector(cfg)
rows = []
for n, (split, seq, i, p) in enumerate(kept):
    img = cv2.imread(p, 0); r = det(img)
    rows.append(dict(split=split, seq=seq, frame=i, file=os.path.basename(p), n_det=len(r["targets"]),
                     conf=max([d.get("conf", d["score"]) for d in r["targets"]], default=0.0), sun=r["sun_flag"], dets=r["targets"]))
    if len(r["targets"]):
        os.makedirs(f"{a.out}/check/{split}", exist_ok=True)
        v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for d in r["targets"]:
            x, y, w, h = d["bbox"]; cv2.rectangle(v, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 0, 255), 2)
        cv2.imwrite(f"{a.out}/check/{split}/{os.path.basename(p)}", v)
    if n % 200 == 0: print(n, flush=True)

for split in ["train", "val", "test"]:
    os.makedirs(f"{a.out}/{split}", exist_ok=True)
    for r in rows:
        if r["split"] == split:
            shutil.copy(f"{a.src}/{r['file']}", f"{a.out}/{split}/{r['file']}")
json.dump(rows, open(f"{a.out}/negatives.json", "w"), indent=0)
for split in ["train", "val", "test"]:
    s = [r for r in rows if r["split"] == split]
    print(split, "кадрів", len(s), "серій", len({r["seq"] for r in s}), "детектор щось бачить:", sum(r["n_det"] > 0 for r in s))
