"""Оцінка YOLO за ТЗ кейсу 4.5 на незалежному тесті.

Правило: влучання = центр рамки в рамці цілі +-3 px; Recall лише для цілей зі сторонами 5-50 px;
FP рахується на кожному наборі окремо. Поріг підбирається на val, на test лише перевіряється.
"""
import os, glob, json, time
import numpy as np, cv2

TOL = 3
CONFS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


# ---------- набори ----------
def yolo_items(root, split, meta=None, only=None):
    """Кадри YOLO-датасету. meta: {stem: dict(scene=..., seq=...)}; only: функція-фільтр по meta."""
    items = []
    for p in sorted(glob.glob(f"{root}/images/{split}/*.jpg")):
        n = os.path.basename(p)[:-4]; m = (meta or {}).get(n, {})
        if only and not only(n, m): continue
        g = []
        lf = f"{root}/labels/{split}/{n}.txt"
        if os.path.exists(lf):
            for l in open(lf).read().split("\n"):
                if l.strip():
                    _, cx, cy, w, h = map(float, l.split()); w, h = w * 640, h * 512
                    g.append((cx * 640 - w / 2, cy * 512 - h / 2, w, h))
        items.append(dict(name=n, seq=m.get("seq", n.split("__")[0]), path=p, gts=g))
    return items


def pad640(img):
    """Кадр будь-якого розміру -> 640x512: більший зменшуємо, менший доповнюємо медіаною (як у skydet)."""
    h, w = img.shape[:2]; s = min(1.0, 640 / w, 512 / h)
    if s < 1: img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA); h, w = img.shape[:2]
    out = np.full((512, 640), int(np.median(img)), np.uint8); x0, y0 = (640 - w) // 2, (512 - h) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out, x0, y0, s


def mask_items(d):
    """Набори skydet (SIRST/IRSTD): images/, masks/, list.txt. Цілі = компоненти маски."""
    items = []
    for n in open(f"{d}/list.txt").read().split():
        p = next(iter(glob.glob(f"{d}/images/{n}.*")), None)
        if p is None: continue
        mp = next(iter(glob.glob(f"{d}/masks/{n}.*")), None)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE); _, x0, y0, s = pad640(img)
        g = []
        if mp:
            k, _, st, _ = cv2.connectedComponentsWithStats((cv2.imread(mp, 0) > 127).astype(np.uint8), 8)
            g = [(st[i, 0] * s + x0, st[i, 1] * s + y0, st[i, 2] * s, st[i, 3] * s) for i in range(1, k)]
        items.append(dict(name=n, seq=os.path.basename(d), path=p, gts=g, pad=True))
    return items


def load(it):
    img = cv2.imread(it["path"], cv2.IMREAD_GRAYSCALE)
    if it.get("pad") or img.shape != (512, 640): img = pad640(img)[0]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ---------- передбачення і метрики ----------
def predict(model, items, batch=16, half=False):
    out = []
    for i in range(0, len(items), batch):
        rs = model.predict([load(it) for it in items[i:i + batch]], imgsz=[512, 640], conf=0.01, max_det=20,
                           verbose=False, batch=batch, **({"half": True} if half else {}))
        for r in rs:
            b = r.boxes
            out.append(np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]).astype(np.float32))
    return out


def ok_size(w, h): return min(w, h) >= 5 and max(w, h) <= 50


def keep(p, conf, topk, margin):
    p = p[p[:, 4] >= conf]
    if margin: p = p[(p[:, 0] >= margin) & (p[:, 1] >= margin) & (p[:, 2] <= 640 - margin) & (p[:, 3] <= 512 - margin)]
    return p[np.argsort(-p[:, 4])][:topk]


def frame_eval(p, gts):
    """-> (кількість FP, множина знайдених цілей)"""
    fp, hit = 0, set()
    for x0, y0, x1, y1, c in p:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        h = [j for j, (x, y, w, hh) in enumerate(gts) if x - TOL <= cx <= x + w + TOL and y - TOL <= cy <= y + hh + TOL]
        fp += not h; hit.update(h)
    return fp, hit


def score(items, preds, cfg):
    conf, topk, margin = cfg["conf"], cfg["topk"], cfg["margin"]
    fps, tp, n, by = [], 0, 0, {}
    for it, p in zip(items, preds):
        fp, hit = frame_eval(keep(p, conf, topk, margin), it["gts"]); fps.append(fp)
        for j, (x, y, w, h) in enumerate(it["gts"]):
            if not ok_size(w, h): continue
            k = "5-8" if max(w, h) <= 8 else "9-16" if max(w, h) <= 16 else "17-32" if max(w, h) <= 32 else "33-50"
            a = by.setdefault(k, [0, 0]); a[1] += 1; n += 1
            if j in hit: a[0] += 1; tp += 1
    fps = np.array(fps) if fps else np.zeros(1)
    return dict(frames=len(items), targets=n, recall=round(tp / n, 4) if n else None,
                fp_per_frame=round(float(fps.mean()), 4), fp_max=int(fps.max()),
                frames_fp_gt1=round(float((fps > 1).mean()), 4), frames_fp_ge1=round(float((fps >= 1).mean()), 4),
                recall_by_size={k: round(a[0] / a[1], 3) for k, a in sorted(by.items())})


def tune(sets, budget, topks):
    """sets: {назва: (items, preds)}. Найбільший Recall на першому наборі, FP <= budget на КОЖНОМУ наборі."""
    best = None
    for topk in topks:
        for margin in [0, 2, 4]:
            for conf in CONFS:
                cfg = dict(conf=conf, topk=topk, margin=margin)
                r = {k: score(it, pr, cfg) for k, (it, pr) in sets.items()}
                if any(x["fp_per_frame"] > budget for x in r.values()): continue
                rec = next(iter(r.values()))["recall"]
                if best is None or rec > best[0] + 1e-9: best = (rec, cfg, r)
    return best


# ---------- розбір помилок ----------
def seq_table(items, preds, cfg):
    t = {}
    for it, p in zip(items, preds):
        fp, hit = frame_eval(keep(p, **cfg), it["gts"])
        a = t.setdefault(it["seq"], dict(sequence=it["seq"], frames=0, fp=0, targets=0, missed=0))
        a["frames"] += 1; a["fp"] += fp
        for j, (x, y, w, h) in enumerate(it["gts"]):
            if ok_size(w, h): a["targets"] += 1; a["missed"] += j not in hit
    return sorted(t.values(), key=lambda a: -(a["fp"] + a["missed"]))


def draw(it, p, cfg, path):
    img = load(it)
    for x, y, w, h in it["gts"]:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 0), 1)
    for x0, y0, x1, y1, c in keep(p, **cfg):
        fp, _ = frame_eval(np.array([[x0, y0, x1, y1, c]]), it["gts"])
        col = (0, 0, 255) if fp else (0, 255, 0)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 1)
        cv2.putText(img, f"{c:.2f}", (int(x0), max(int(y0) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.imwrite(path, img)


def gallery(items, preds, cfg, out, n=12):
    """n кадрів із найупевненішими FP і n пропусків, не більше 2 кадрів з однієї серії."""
    os.makedirs(out, exist_ok=True); fps, miss = [], []
    for i, (it, p) in enumerate(zip(items, preds)):
        k = keep(p, **cfg); fp, hit = frame_eval(k, it["gts"])
        if fp: fps.append((-float(k[:, 4].max()), i))
        if any(ok_size(w, h) and j not in hit for j, (x, y, w, h) in enumerate(it["gts"])): miss.append((i,))
    html = ["<html><meta charset='utf-8'><body style='background:#111;color:#eee;font:14px sans-serif'>",
            "<p>Блакитна рамка: розмітка. Зелена: влучання. Червона: хибне спрацювання.</p>"]
    for title, lst in [("Хибні спрацювання", sorted(fps)), ("Пропуски", miss[::max(1, len(miss) // (n * 4))])]:
        html.append(f"<h2>{title}</h2>"); per, k = {}, 0
        for e in lst:
            it = items[e[-1]]
            if per.get(it["seq"], 0) >= 2: continue
            per[it["seq"]] = per.get(it["seq"], 0) + 1; k += 1
            f = f"{'fp' if title[0] == 'Х' else 'miss'}_{k:02d}.jpg"; draw(it, preds[e[-1]], cfg, f"{out}/{f}")
            html.append(f"<div style='display:inline-block;margin:6px'><img src='{f}' width=480><br>{it['name']}</div>")
            if k >= n: break
    open(f"{out}/gallery.html", "w").write("\n".join(html))


# ---------- швидкість ----------
def speed(model, items, cfg, half=False, n=200):
    """Увесь конвеєр на кадрі в пам'яті, batch 1: підготовка, модель, NMS, наші фільтри."""
    imgs = [load(it) for it in items[:: max(1, len(items) // 50)][:50]]
    kw = {"half": True} if half else {}
    for i in range(20): model.predict(imgs[i % len(imgs)], imgsz=[512, 640], conf=cfg["conf"], verbose=False, **kw)
    t = []
    for i in range(n):
        t0 = time.perf_counter()
        b = model.predict(imgs[i % len(imgs)], imgsz=[512, 640], conf=cfg["conf"], verbose=False, **kw)[0].boxes
        keep(np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]), **cfg)
        t.append((time.perf_counter() - t0) * 1000)
    return dict(ms_mean=round(float(np.mean(t)), 2), ms_p95=round(float(np.percentile(t, 95)), 2), fps=round(1000 / float(np.mean(t)), 1))
