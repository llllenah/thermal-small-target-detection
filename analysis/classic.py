"""Classical baseline: white top-hat candidates + physical post-filters.
Measures Recall (targets 5..50 px) and FP per frame after each added stage, plus time per 640x512 frame."""
import os, json, time, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("eda_loaders", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_loaders.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)

K_TOPHAT = 31
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_TOPHAT, K_TOPHAT))
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def local_contrast(im, x, y, w, h):
    H, W = im.shape
    x1, y1 = min(x + w, W), min(y + h, H)
    t = im[y:y1, x:x1].astype(np.float32)
    p = max(w, h, 3)
    X0, Y0, X1, Y1 = max(x - p, 0), max(y - p, 0), min(x1 + p, W), min(y1 + p, H)
    ring = im[Y0:Y1, X0:X1].astype(np.float32)
    m = np.ones_like(ring, bool); m[y - Y0:y1 - Y0, x - X0:x1 - X0] = False
    r = ring[m]
    return (t.mean() - r.mean()) / (r.std() + 1.0)

def run(im, use_clahe=False):
    """returns dict stage -> list of (cx, cy)"""
    t0 = time.perf_counter()
    src = CLAHE.apply(im) if use_clahe else im
    # sun / saturation mask
    sat = (im >= 250).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    sun = np.zeros_like(sat)
    for i in range(1, n):
        if st[i, 4] >= 200:
            sun[lab == i] = 1
    if sun.any():
        sun = cv2.dilate(sun, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    th = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, SE)
    thr = th.mean() + 4 * th.std()
    bw = (th > max(thr, 10)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        cands.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), a=int(a), cx=float(cen[i, 0]), cy=float(cen[i, 1])))
    t_det = time.perf_counter()
    out = {"0 top-hat + поріг": cands}
    c1 = [c for c in cands if not sun[int(c["cy"]), int(c["cx"])]]
    out["1 + маска сонця"] = c1
    c2 = [c for c in c1 if c["a"] >= 6 and max(c["w"], c["h"]) <= 50]
    out["2 + розмір"] = c2
    c3 = [c for c in c2 if max(c["w"], c["h"]) / max(1, min(c["w"], c["h"])) <= 3]
    out["3 + форма"] = c3
    for c in c3:
        c["scr"] = local_contrast(im, c["x"], c["y"], c["w"], c["h"])
    c4 = [c for c in c3 if c["scr"] >= 3]
    out["4 + локальний контраст"] = c4
    c5 = sorted(c4, key=lambda c: -c["scr"])[:5]
    out["5 + top-5"] = c5
    t_end = time.perf_counter()
    return out, (t_end - t0) * 1000

def evaluate(items, use_clahe=False, limit=None):
    stages = None; tp = {}; fp = {}; ngt = 0; nfr = 0; times = []
    for it in items[:limit] if limit else items:
        im0 = L.gray(it["path"])
        im, bx, s = L.to640(im0, it["boxes"])
        res, ms = run(im, use_clahe); times.append(ms)
        nfr += 1
        gts = [(x - 3, y - 3, x + w + 3, y + h + 3, min(w, h) >= 5 and max(w, h) <= 50) for x, y, w, h, c in bx]
        ngt += sum(g[4] for g in gts)
        for k, cs in res.items():
            hit = set(); f = 0
            for c in cs:
                m = [j for j, g in enumerate(gts) if g[0] <= c["cx"] <= g[2] and g[1] <= c["cy"] <= g[3]]
                if m:
                    hit.update(j for j in m if gts[j][4])
                else:
                    f += 1
            tp[k] = tp.get(k, 0) + len(hit); fp[k] = fp.get(k, 0) + f
    rows = [dict(stage=k, recall=(tp[k] / ngt if ngt else None), fp_per_frame=fp[k] / nfr) for k in tp]
    return dict(frames=nfr, targets=ngt, ms_mean=float(np.mean(times)), ms_p95=float(np.percentile(times, 95)), rows=rows)

if __name__ == "__main__":
    S = L.S
    hit = [i for i in L.load_hit() if i["split"] == "test"]
    flir = L.load_flir()
    sir = L.load_sirst()
    sir_t = [i for i in sir if i["boxes"]]
    sir_bg = [i for i in sir if not i["boxes"]]
    res = {}
    for name, items in [("HIT-UAV (test, дрон)", hit), ("FLIR (авто)", flir), ("SIRST-v2 (цілі)", sir_t), ("SIRST-v2 (фон без цілей)", sir_bg)]:
        for cl in [False, True]:
            key = name + (" +CLAHE" if cl else "")
            res[key] = evaluate(items, cl)
            r = res[key]
            print(f"\n{key}: frames {r['frames']} targets {r['targets']} time {r['ms_mean']:.1f} ms (p95 {r['ms_p95']:.1f})")
            for row in r["rows"]:
                rc = "-" if row["recall"] is None else f"{row['recall']*100:.1f}%"
                print(f"   {row['stage']:<28} recall {rc:>7}  FP/кадр {row['fp_per_frame']:.2f}")
    json.dump(res, open(f"{S}/eda_out/classic.json", "w"), ensure_ascii=False, indent=1)
