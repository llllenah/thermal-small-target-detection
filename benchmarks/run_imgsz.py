"""YOLO з imgsz 640 / 960 / 1280 (кадр 640x512 збільшується до 640x512, 960x768, 1280x1024), TensorRT FP16, окремі engine.
Пороги ті самі: ty=0.15, ts=0.3, a=0.25, top-1. Recall, FP на кадр, мс на кадр -> imgsz.csv
"""
import os, sys, csv, time, pickle
import numpy as np

W = "/workspace"; OUT = f"{W}/results/extra"; CACHE = f"{OUT}/cache"; ENG = f"{OUT}/engines"
os.makedirs(ENG, exist_ok=True)
sys.path[:0] = [f"{W}/thermal_eda/yolo", f"{W}/thermal_eda/skydet"]

import torch, tensorrt as trt, tensorrt_libs  # noqa: F401
import onnxruntime as ort
ort.preload_dlls()
from ultralytics import YOLO
import eval_tz as E, fuse as F
from detector import SkyDetector, load_cfg

CFG = dict(ty=0.15, ts=0.3, a=0.25)
SIZES = {640: [512, 640], 960: [768, 960], 1280: [1024, 1280]}
SETS = {f"sirst_{k}": E.mask_items(f"{W}/{k}") for k in ["sky_targets", "sun_real", "no_targets"]}
N = 1000
NONE = np.zeros((0, 5), np.float32)

SCFG = load_cfg(f"{W}/skydet/skydet_config.json"); SCFG.update(model=f"{W}/skydet/adgf_lite.onnx", topk=5, lr_thr=0.0)
SKY = SkyDetector(SCFG, providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                                  "trt_engine_cache_path": f"{OUT}/trt_cache"}),
                                   "CUDAExecutionProvider", "CPUExecutionProvider"])
assert SKY.provider == "TensorrtExecutionProvider"
PS = {k: pickle.load(open(f"{CACHE}/skydet_trt_fp16_{k}.pkl", "rb")) for k in SETS}     # ті самі передбачення skydet, що в run_extra.py


def engine(sz):
    if sz == 640: return f"{W}/best.engine"
    e = f"{ENG}/best_{sz}.engine"
    if not os.path.exists(e):
        pt = f"{ENG}/best_{sz}.pt"
        if not os.path.exists(pt): os.symlink(f"{W}/best.pt", pt)
        YOLO(pt).export(format="engine", imgsz=SIZES[sz], batch=1, device=0, half=True)
    return e


def predict(model, items, imgsz):                   # як E.predict, але з іншим imgsz; рамки в координатах кадру 640x512
    out = []
    for it in items:
        b = model.predict(E.load(it), imgsz=imgsz, conf=0.01, max_det=20, verbose=False)[0].boxes
        out.append(np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]).astype(np.float32))
    return out


def bench(run, imgs, n=N, warm=50):
    for i in range(warm): run(imgs[i % len(imgs)])
    t = []
    for i in range(n):
        t0 = time.perf_counter(); run(imgs[i % len(imgs)]); t.append((time.perf_counter() - t0) * 1000)
    return dict(ms_mean=round(float(np.mean(t)), 2), ms_p95=round(float(np.percentile(t, 95)), 2), fps=round(1000 / float(np.mean(t)), 1))


allit = [x for it in SETS.values() for x in it]
imgs = [E.load(x) for x in allit[:: max(1, len(allit) // 50)][:50]]
rows = []
for sz, imgsz in SIZES.items():
    ym = YOLO(engine(sz), task="detect")
    PY = {}
    for k, it in SETS.items():
        p = f"{CACHE}/yolo_trt_fp16_imgsz{sz}_{k}.pkl"
        if os.path.exists(p): PY[k] = pickle.load(open(p, "rb"))
        else: PY[k] = predict(ym, it, imgsz); pickle.dump(PY[k], open(p, "wb"))

    def run_yolo(img):
        bx = ym.predict(img, imgsz=imgsz, conf=CFG["ty"], verbose=False)[0].boxes
        return E.keep(np.hstack([bx.xyxy.cpu().numpy(), bx.conf.cpu().numpy()[:, None]]), CFG["ty"], 1, 0)

    def run_pipe(img):
        bx = ym.predict(img, imgsz=imgsz, conf=CFG["ty"], verbose=False)[0].boxes
        py = np.hstack([bx.xyxy.cpu().numpy(), bx.conf.cpu().numpy()[:, None]])
        return F.fuse(py, F.sky_pred(SKY, img[:, :, 0]), CFG["ty"], CFG["ts"], CFG["a"])

    sp = {"yolo": bench(run_yolo, imgs), "yolo+skydet": bench(run_pipe, imgs)}
    for k, it in SETS.items():
        no = [NONE] * len(it)
        for system, r in [("yolo", F.score(it, PY[k], no, CFG)), ("yolo+skydet", F.score(it, PY[k], PS[k], CFG))]:
            rows.append(dict(imgsz=sz, input=f"{imgsz[1]}x{imgsz[0]}", system=system, set=k, frames=r["frames"], targets=r["targets"],
                             recall=r["recall"], fp_per_frame=r["fp_per_frame"], fp_max=r["fp_max"],
                             **{f"recall_{s}px": r["recall_by_size"].get(s) for s in ["5-8", "9-16", "17-32", "33-50"]},
                             **sp[system]))
            print(rows[-1], flush=True)
with open(f"{OUT}/imgsz.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("imgsz.csv", len(rows), "рядків")
