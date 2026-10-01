"""Case 4.5 metrics for a BasicIRSTD segmentation model.

For every test subset it reports what the TZ asks for, not pixel IoU:
  * Recall over GT targets 5..50 px (and over all GT targets),
  * FP per frame (mean and 95th percentile),
  * FPS of the whole pipeline on 640x512 (normalise -> model -> threshold -> post-filters).

A prediction = connected component of (prob > thr). It is a TP when its centroid falls into a GT
target box expanded by `tol` px. Predictions that hit only GT targets outside 5..50 px are ignored
(neither TP nor FP), as agreed in CLAUDE.md.

Post-filters (all optional): sun/saturation mask, component size limits, top-K by peak probability.

Run from the BasicIRSTD folder:
  python eval_thermal.py --model DNANet --ckpt log/THERMAL-MIX/DNANet_100.pth.tar \
      --subsets THERMAL-MIX-FT-hit THERMAL-MIX-FT-flir THERMAL-MIX-FT-sirst THERMAL-MIX-FT-sirst_background THERMAL-MIX-FT-sun \
      --mean 100.0 --std 30.0 --thr 0.5 --topk 5 --sun --json results.json
Add --sweep to try several thresholds / top-K on one subset list (use the validation set: THERMAL-MIX).
"""
import os, sys, time, json, argparse
import numpy as np, cv2, torch

sys.path.insert(0, os.getcwd())


def build(model_name, ckpt, device):
    from net import Net
    net = Net(model_name=model_name, mode="test")
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    return net.to(device).eval()


def gt_targets(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        out.append(dict(x=x, y=y, w=w, h=h, in_scope=bool(min(w, h) >= 5 and max(w, h) <= 50)))
    return out


def sun_mask(img, level=250, min_area=200, grow=31):
    sat = (img >= level).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    m = np.zeros_like(sat)
    for i in range(1, n):
        if st[i, 4] >= min_area:
            m[lab == i] = 1
    if m.any():
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return m


def detections(prob, img, thr, use_sun, min_side, max_side, topk):
    bw = (prob > thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    sm = sun_mask(img) if use_sun else None
    dets = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        cx, cy = cen[i]
        if sm is not None and sm[int(cy), int(cx)]:
            continue
        if max(w, h) < min_side or max(w, h) > max_side:
            continue
        dets.append(dict(cx=float(cx), cy=float(cy), x=int(x), y=int(y), w=int(w), h=int(h),
                         score=float(prob[lab == i].max())))
    dets.sort(key=lambda d: -d["score"])
    return dets[:topk] if topk > 0 else dets


def match(dets, gts, tol):
    tp = set(); fp = 0; ign = 0
    for d in dets:
        hit = [j for j, g in enumerate(gts) if g["x"] - tol <= d["cx"] <= g["x"] + g["w"] + tol and g["y"] - tol <= d["cy"] <= g["y"] + g["h"] + tol]
        if not hit:
            fp += 1
        elif any(gts[j]["in_scope"] for j in hit):
            tp.update(j for j in hit if gts[j]["in_scope"])
        else:
            ign += 1
    hit_all = set()
    for d in dets:
        hit_all.update(j for j, g in enumerate(gts) if g["x"] - tol <= d["cx"] <= g["x"] + g["w"] + tol and g["y"] - tol <= d["cy"] <= g["y"] + g["h"] + tol)
    return len(tp), fp, len(hit_all)


def pad32(a):
    h, w = a.shape
    H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    return np.pad(a, ((0, H - h), (0, W - w))), (h, w)


@torch.no_grad()
def predict(net, img, mean, std, device, half):
    x, (h, w) = pad32((img.astype(np.float32) - mean) / std)
    t = torch.from_numpy(x)[None, None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=half and device == "cuda"):
        p = net(t)
    if isinstance(p, (list, tuple)):
        p = p[-1]
    return p[0, 0, :h, :w].float().cpu().numpy()


def run_subset(net, ds_dir, name, a, device, cache=None):
    ids = open(f"{ds_dir}/{name}/img_idx/test_{name}.txt").read().split()
    tp = n_in = fp_tot = hit_all = n_all = 0
    fps_per_frame = []
    for iid in ids:
        img = cv2.imread(f"{ds_dir}/{name}/images/{iid}.png", 0)
        gts = gt_targets(cv2.imread(f"{ds_dir}/{name}/masks/{iid}.png", 0))
        key = (name, iid)
        if cache is not None and key in cache:
            prob = cache[key]
        else:
            prob = predict(net, img, a.mean, a.std, device, a.half)
            if cache is not None:
                cache[key] = prob
        dets = detections(prob, img, a.thr, a.sun, a.min_side, a.max_side, a.topk)
        t, f, h = match(dets, gts, a.tol)
        tp += t; fp_tot += f; hit_all += h
        n_in += sum(g["in_scope"] for g in gts); n_all += len(gts)
        fps_per_frame.append(f)
    return dict(frames=len(ids), targets_5_50=n_in,
                recall_5_50=round(tp / n_in, 4) if n_in else None,
                recall_all=round(hit_all / n_all, 4) if n_all else None,
                fp_per_frame=round(fp_tot / max(1, len(ids)), 3),
                fp_p95=float(np.percentile(fps_per_frame, 95)) if fps_per_frame else 0.0)


@torch.no_grad()
def bench_fps(net, a, device, n=200):
    img = (np.random.rand(512, 640) * 255).astype(np.uint8)
    for _ in range(10):
        detections(predict(net, img, a.mean, a.std, device, a.half), img, a.thr, a.sun, a.min_side, a.max_side, a.topk)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        prob = predict(net, img, a.mean, a.std, device, a.half)
        detections(prob, img, a.thr, a.sun, a.min_side, a.max_side, a.topk)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / n * 1000
    return dict(ms_per_frame=round(ms, 2), fps=round(1000 / ms, 1), device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu", fp16=a.half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DNANet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_dir", default="./datasets")
    ap.add_argument("--subsets", nargs="+", required=True)
    ap.add_argument("--mean", type=float, required=True)
    ap.add_argument("--std", type=float, required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--sun", action="store_true")
    ap.add_argument("--min_side", type=int, default=2)
    ap.add_argument("--max_side", type=int, default=80)
    ap.add_argument("--topk", type=int, default=0, help="0 = keep all")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--fps", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build(a.model, a.ckpt, device)
    out = {"model": a.model, "ckpt": a.ckpt, "settings": {k: getattr(a, k) for k in ["thr", "tol", "sun", "min_side", "max_side", "topk", "half"]}}

    if a.sweep:
        cache, rows = {}, []
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for topk in [0, 5, 3]:
                a.thr, a.topk = thr, topk
                r = {s: run_subset(net, a.dataset_dir, s, a, device, cache) for s in a.subsets}
                worst_fp = max(v["fp_per_frame"] for v in r.values())
                rec = np.mean([v["recall_5_50"] for v in r.values() if v["recall_5_50"] is not None])
                rows.append(dict(thr=thr, topk=topk, mean_recall_5_50=round(float(rec), 4), worst_fp_per_frame=worst_fp))
                print(rows[-1], flush=True)
        ok = [r for r in rows if r["worst_fp_per_frame"] <= 0.7]
        out["sweep"] = rows
        out["best_under_0.7fp"] = max(ok, key=lambda r: r["mean_recall_5_50"]) if ok else None
        print("best:", out["best_under_0.7fp"])
    else:
        out["subsets"] = {}
        for s in a.subsets:
            out["subsets"][s] = run_subset(net, a.dataset_dir, s, a, device)
            print(s, out["subsets"][s], flush=True)
    if a.fps:
        out["fps"] = bench_fps(net, a, device)
        print("FPS:", out["fps"])
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
