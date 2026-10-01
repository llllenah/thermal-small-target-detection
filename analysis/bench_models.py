"""One table for any segmentation IRSTD model: object-level Recall / FP per frame (ТЗ metrics) + speed.

  python bench_models.py --model adgf_lite --weights ADGFNet/checkpoints/IRSTD-1K/ADGFNetLite.pth.tar \
      --images IRSTD1k_Img --masks IRSTD1k_Label --list test.txt --mean 87.47 --std 39.72

A detection = connected component of sigmoid(output) > thr. TP when its centre lies inside a GT target box
expanded by `tol` px. GT targets outside 5..50 px are "ignored": hitting them is neither TP nor FP.
"""
import os, sys, time, json, argparse
import numpy as np, cv2, torch


def load_model(name, weights, repo):
    sys.path.insert(0, repo)
    if name in ("adgf", "adgf_lite"):
        from model import ADGFNet, ADGFNetLite
        net = ADGFNetLite() if name == "adgf_lite" else ADGFNet()
        sd = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
        net.load_state_dict(sd)
        return net.eval(), True          # returns logits -> needs sigmoid
    if name == "mshnet":
        from model.MSHNet import MSHNet
        net = MSHNet(3)
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd.get("net", sd)) if isinstance(sd, dict) else sd
        net.load_state_dict(sd)
        class Wrap(torch.nn.Module):          # 3-channel ImageNet-normalised input, forward(x, warm_flag)
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x):
                x = (x.repeat(1, 3, 1, 1) / 255.0 - torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
                return s.m(x, True)[1]
        return Wrap(net).eval(), True
    raise ValueError(name)


def targets(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    return [dict(x=st[i, 0], y=st[i, 1], w=st[i, 2], h=st[i, 3],
                 ok=bool(min(st[i, 2], st[i, 3]) >= 5 and max(st[i, 2], st[i, 3]) <= 50)) for i in range(1, n)]


def detect(prob, thr, topk, min_area):
    n, lab, st, cen = cv2.connectedComponentsWithStats((prob > thr).astype(np.uint8), 8)
    d = [dict(cx=cen[i, 0], cy=cen[i, 1], s=float(prob[lab == i].max())) for i in range(1, n) if st[i, 4] >= min_area]
    d.sort(key=lambda z: -z["s"])
    return d[:topk] if topk else d


def score(dets, gts, tol=3):
    tp, fp = set(), 0
    for d in dets:
        hit = [j for j, g in enumerate(gts) if g["x"] - tol <= d["cx"] <= g["x"] + g["w"] + tol and g["y"] - tol <= d["cy"] <= g["y"] + g["h"] + tol]
        if not hit:
            fp += 1
        tp.update(j for j in hit if gts[j]["ok"])
    return len(tp), fp


def pad32(a):
    h, w = a.shape
    return np.pad(a, ((0, (-h) % 32), (0, (-w) % 32))), (h, w)


DEV, HALF = "cpu", False


@torch.no_grad()
def infer(net, sig, img, mean, std):
    x, (h, w) = pad32((img.astype(np.float32) - mean) / std)   # for mshnet pass --mean 0 --std 1 (raw 0..255)
    t = torch.from_numpy(x)[None, None].to(DEV)
    with torch.autocast("cuda", dtype=torch.float16, enabled=HALF and DEV == "cuda"):
        p = net(t)
    p = p[-1] if isinstance(p, (list, tuple)) else p
    p = torch.sigmoid(p) if sig else p
    return p[0, 0, :h, :w].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--repo", default="."); ap.add_argument("--images", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--list", required=True); ap.add_argument("--mean", type=float, required=True); ap.add_argument("--std", type=float, required=True)
    ap.add_argument("--min_area", type=int, default=1); ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--half", action="store_true")
    a = ap.parse_args()
    global DEV, HALF
    DEV, HALF = a.device, a.half
    torch.set_num_threads(os.cpu_count())
    net, sig = load_model(a.model, a.weights, a.repo)
    net = net.to(DEV)
    ids = open(a.list).read().split()
    probs = []
    for i in ids:
        img = cv2.imread(f"{a.images}/{i}.png", 0); m = cv2.imread(f"{a.masks}/{i}.png", 0)
        probs.append((infer(net, sig, img, a.mean, a.std), targets(m)))
    n_ok = sum(g["ok"] for _, gts in probs for g in gts)
    rows = []
    for thr in [0.3, 0.5, 0.7]:
        for topk in [0, 3, 1]:
            tp = fp = 0
            for p, gts in probs:
                t, f = score(detect(p, thr, topk, a.min_area), gts); tp += t; fp += f
            rows.append(dict(thr=thr, topk=topk, recall_5_50=round(tp / n_ok, 3), fp_per_frame=round(fp / len(ids), 3)))
            print(rows[-1], flush=True)
    # speed on a 640x512 frame, whole pipeline, batch 1
    img = (np.random.rand(512, 640) * 255).astype(np.uint8)
    for _ in range(3): detect(infer(net, sig, img, a.mean, a.std), 0.5, 3, 1)
    n = 100 if DEV == "cuda" else 20
    if DEV == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): detect(infer(net, sig, img, a.mean, a.std), 0.5, 3, 1)
    if DEV == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / n * 1000
    params = sum(p.numel() for p in net.parameters()) / 1e6
    res = dict(model=a.model, frames=len(ids), targets_5_50=n_ok, params_M=round(params, 3),
               device=torch.cuda.get_device_name(0) if DEV == "cuda" else f"cpu x{os.cpu_count()}", fp16=HALF,
               ms_640x512=round(ms, 2), fps_640x512=round(1000 / ms, 1), rows=rows)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
