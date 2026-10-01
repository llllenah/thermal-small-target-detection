"""Оцінка за ТЗ на кожному наборі окремо: Recall цілей 5-50 px, FP на кадр, максимум FP на кадрі, FPS.

  python evaluate.py --data testsets --tune          # підбір thr/top-K на val_* -> config.json
  python evaluate.py --data testsets --cfg config.json --ablation --json results.json

Влучання: центр детекції в рамці цілі +-3 px. Цілі <5 або >50 px ігноруються (влучання в них не TP і не FP).
"""
import os, json, time, argparse, itertools
import numpy as np, cv2
from detector import SkyDetector, load_cfg, fit, features

TOL = 3
VAL = ["val_irstd", "val_sun", "val_bg"]
TEST = ["sky_targets", "no_targets", "sun_synth", "sun_real"]


def targets(mask):
    n, _, st, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    return [(st[i, 0], st[i, 1], st[i, 2], st[i, 3], min(st[i, 2], st[i, 3]) >= 5 and max(st[i, 2], st[i, 3]) <= 50) for i in range(1, n)]


def cache(det, root):
    """Прогін моделі один раз; ймовірності зберігаємо як uint8, щоб потім швидко перебирати фільтри."""
    names = open(f"{root}/list.txt").read().split()
    cf = f"{root}/prob_{os.path.basename(det.cfg['model'])}_{det.provider[:4]}.npy"
    probs = np.load(cf) if os.path.exists(cf) else None
    out = []
    for i, n in enumerate(names):
        img, _, hw = fit(cv2.imread(f"{root}/images/{n}.png", 0))
        m, _, _ = fit(cv2.imread(f"{root}/masks/{n}.png", 0), pad=0)
        out.append((img, probs[i] if probs is not None else (det.prob(img) * 255).astype(np.uint8), hw, targets(m)))
    if probs is None:
        np.save(cf, np.stack([o[1] for o in out]))
    return out


def score(det, data, cfg):
    det.cfg = {**det.cfg, **cfg}
    tp = n_ok = 0; fps = []
    for img, p8, hw, gts in data:
        dets, _ = det.post(p8.astype(np.float32) / 255, img, hw)
        hit, fp = set(), 0
        for d in dets:
            cx, cy = d["center"]
            h = [j for j, (x, y, w, hh, _) in enumerate(gts) if x - TOL <= cx <= x + w + TOL and y - TOL <= cy <= y + hh + TOL]
            fp += not h; hit.update(j for j in h if gts[j][4])
        tp += len(hit); n_ok += sum(g[4] for g in gts); fps.append(fp)
    fps = np.array(fps)
    return dict(frames=len(data), targets=int(n_ok), recall=round(float(tp / n_ok), 3) if n_ok else None,
                fp_per_frame=round(float(fps.mean()), 3), fp_max=int(fps.max()), frames_fp_gt1=round(float((fps > 1).mean()), 3))


def fit_lr(det, data):
    """Логістична регресія: справжня пляма (влучила в ціль) проти хибної, на всіх детекціях val-наборів."""
    from sklearn.linear_model import LogisticRegression
    det.cfg = {**det.cfg, "thr": 0.3, "min_side": 1, "use_lr": False, "use_topk": False}
    X, Y = [], []
    for s in VAL:
        for img, p8, hw, gts in data[s]:
            for d in det.post(p8.astype(np.float32) / 255, img, hw)[0]:
                cx, cy = d["center"]
                X.append(features(img, d)); Y.append(any(x - TOL <= cx <= x + w + TOL and y - TOL <= cy <= y + h + TOL for x, y, w, h, _ in gts))
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(np.array(X), np.array(Y))
    return [round(float(v), 4) for v in list(lr.coef_[0]) + list(lr.intercept_)]


def speed(det, img, n=100):
    for _ in range(10): det(img)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); det(img); t.append((time.perf_counter() - t0) * 1000)
    return dict(provider=det.provider, ms_mean=round(float(np.mean(t)), 2), ms_p95=round(float(np.percentile(t, 95)), 2), fps=round(1000 / np.mean(t), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--cfg"); ap.add_argument("--model"); ap.add_argument("--backend")
    ap.add_argument("--tune", action="store_true"); ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--json", default="results.json"); ap.add_argument("--fp_budget", type=float, default=0.7, help="FP на кадр на кожному val-наборі")
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    if a.model: cfg["model"] = a.model
    if a.backend: cfg["backend"] = a.backend
    if not os.path.isabs(cfg["model"]) and not os.path.exists(cfg["model"]):
        cfg["model"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["model"])
    det = SkyDetector(cfg)
    sets = VAL if a.tune else TEST
    data = {s: cache(det, f"{a.data}/{s}") for s in sets}; print("model done", flush=True)

    if a.tune:
        cfg["lr_coef"] = fit_lr(det, data); print("lr_coef", cfg["lr_coef"], flush=True)
        best = None
        for thr, k, ms, lt in itertools.product([0.3, 0.5], [3, 5], [1, 2], [0, 0.2, 0.3, 0.4, 0.5, 0.6]):
            r = {s: score(det, data[s], dict(thr=thr, topk=k, use_topk=True, min_side=ms, lr_coef=cfg["lr_coef"], lr_thr=lt, use_lr=lt > 0)) for s in VAL}
            ok = all(x["fp_per_frame"] <= a.fp_budget for x in r.values())
            rec = round(float(np.mean([x["recall"] for x in r.values() if x["recall"] is not None])), 3); fpw = max(x["fp_per_frame"] for x in r.values())
            print(thr, k, ms, lt, ok, {s: (x["recall"], x["fp_per_frame"]) for s, x in r.items()}, flush=True)
            if ok and (best is None or (rec, -fpw) > (best[0], -best[6])):
                best = (rec, thr, k, ms, lt, r, fpw)
        cfg.update(thr=best[1], topk=best[2], use_topk=True, min_side=best[3], lr_thr=best[4], use_lr=best[4] > 0,
                   model=os.path.basename(cfg["model"]))
        json.dump(cfg, open("config.json", "w"), indent=1)
        print("BEST", best); return

    res = {"config": cfg, "sets": {s: score(det, data[s], cfg) for s in TEST}}
    if a.ablation:          # внесок кожного фільтра
        steps = [("модель", dict(use_sun=False, use_size=False, use_lr=False, use_topk=False)), ("+ сонце", dict(use_sun=True, use_size=False, use_lr=False, use_topk=False)),
                 ("+ розмір", dict(use_sun=True, use_size=True, use_lr=False, use_topk=False)), ("+ фільтр плям", dict(use_sun=True, use_size=True, use_lr=True, use_topk=False)),
                 ("+ top-K", dict(use_sun=True, use_size=True, use_lr=True, use_topk=True))]
        res["ablation"] = {name: {s: score(det, data[s], {**cfg, **c}) for s in TEST} for name, c in steps}
        det.cfg = cfg
    res["speed"] = speed(det, data[TEST[0]][0][0])
    json.dump(res, open(a.json, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["sets"], indent=1), res["speed"])


if __name__ == "__main__":
    main()
