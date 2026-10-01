"""YOLO + skydet (ADGFNet-Lite): об'єднання детекцій і підбір порогів за ТЗ.

YOLO добре знаходить дрони 9-50 px, skydet знаходить точкові плями (птахи, далекі дрони).
Кандидати: рамки YOLO з conf >= ty і плями skydet з conf >= ts, яких немає всередині рамок YOLO.
Бал плями skydet множиться на a. Лишаємо top-1 за балом: на кадрі не більше 1 FP.
"""
import time
import numpy as np
import eval_tz as E

TYS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
TSS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
AS = [0.25, 0.5, 1.0, 2.0]


def sky_pred(det, img_gray):
    """Результат SkyDetector -> масив [x0, y0, x1, y1, conf]."""
    r = det(img_gray)["targets"]
    return np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3],
                      d.get("conf", d["score"])] for d in r], np.float32).reshape(-1, 5)


def sky_predict(det, items):
    return [sky_pred(det, E.load(it)[:, :, 0]) for it in items]


def fuse(py, ps, ty, ts, a, topk=1):
    y = py[py[:, 4] >= ty]
    s = ps[ps[:, 4] >= ts].copy()
    if len(y) and len(s):
        cx, cy = (s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2
        inside = ((cx[:, None] >= y[None, :, 0] - 3) & (cx[:, None] <= y[None, :, 2] + 3) &
                  (cy[:, None] >= y[None, :, 1] - 3) & (cy[:, None] <= y[None, :, 3] + 3)).any(1)
        s = s[~inside]
    s[:, 4] *= a
    c = np.vstack([y, s])
    return c[np.argsort(-c[:, 4])][:topk]


def fused_list(PY, PS, cfg):
    return [fuse(py, ps, cfg["ty"], cfg["ts"], cfg["a"]) for py, ps in zip(PY, PS)]


def score(items, PY, PS, cfg):
    # після fuse рамки вже відфільтровані: conf 0 і top-1 нічого не змінюють
    return E.score(items, fused_list(PY, PS, cfg), dict(conf=0, topk=1, margin=0))


def tune(sets, budget, target_sets):
    """sets: {назва: (items, PY, PS)}. Найбільший середній Recall на target_sets, FP <= budget на кожному наборі."""
    best = None
    for ty in TYS:
        for ts in TSS:
            for a in AS:
                cfg = dict(ty=ty, ts=ts, a=a)
                r = {}
                for k, (it, py, ps) in sets.items():
                    r[k] = score(it, py, ps, cfg)
                    if r[k]["fp_per_frame"] > budget: break
                else:
                    rec = float(np.mean([r[k]["recall"] for k in target_sets]))
                    if best is None or rec > best[0] + 1e-9: best = (rec, cfg, r)
    return best


def speed(model, det, items, cfg, n=200):
    """Увесь конвеєр, batch 1: YOLO + skydet + об'єднання."""
    imgs = [E.load(it) for it in items[:: max(1, len(items) // 50)][:50]]
    def run(img):
        b = model.predict(img, imgsz=[512, 640], conf=cfg["ty"], verbose=False)[0].boxes
        py = np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]])
        return fuse(py, sky_pred(det, img[:, :, 0]), cfg["ty"], cfg["ts"], cfg["a"])
    for i in range(20): run(imgs[i % len(imgs)])
    t = []
    for i in range(n):
        t0 = time.perf_counter(); run(imgs[i % len(imgs)]); t.append((time.perf_counter() - t0) * 1000)
    return dict(ms_mean=round(float(np.mean(t)), 2), ms_p95=round(float(np.percentile(t, 95)), 2), fps=round(1000 / float(np.mean(t)), 1))
