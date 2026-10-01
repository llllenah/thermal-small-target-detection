"""RTX 4090: швидкість (YOLO TRT FP16, skydet CUDA FP32 / TRT FP16, увесь конвеєр), точність skydet FP32 проти TRT FP16,
криві Recall / FP на кадр. Логіка злиття з thermal_eda/yolo/fuse.py, метрика з thermal_eda/yolo/eval_tz.py.

  python run_extra.py            # усе: передбачення (кеш), accuracy.json, curves.csv, speed.json
"""
import os, sys, json, time, pickle, csv
import numpy as np

W = "/workspace"; OUT = f"{W}/results/extra"; CACHE = f"{OUT}/cache"
os.makedirs(CACHE, exist_ok=True)
sys.path[:0] = [f"{W}/thermal_eda/yolo", f"{W}/thermal_eda/skydet"]

import torch, tensorrt as trt, tensorrt_libs  # noqa: F401  (libnvinfer.so.10 для ONNX Runtime)
import onnxruntime as ort
ort.preload_dlls()                               # CUDA/cuDNN з PyTorch
from ultralytics import YOLO
import eval_tz as E, fuse as F
from detector import SkyDetector, load_cfg

CFG = dict(ty=0.15, ts=0.3, a=0.25)              # пороги злиття, top-1
SETS = {f"sirst_{k}": E.mask_items(f"{W}/{k}") for k in ["sky_targets", "sun_real", "no_targets"]}
N = 1000
THRS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95]
NONE = np.zeros((0, 5), np.float32)


def no(P): return [NONE] * len(P)


def cached(name, fn):
    p = f"{CACHE}/{name}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r = fn(); pickle.dump(r, open(p, "wb")); return r


# ---------- моделі ----------
eng = f"{W}/best.engine"
if not os.path.exists(eng):
    YOLO(f"{W}/best.pt").export(format="engine", imgsz=[512, 640], batch=1, device=0, half=True)
ym = YOLO(eng, task="detect")
ypt = YOLO(f"{W}/best.pt")

SCFG = load_cfg(f"{W}/skydet/skydet_config.json")
# як в ensemble_test: більше кандидатів, поріг ts застосовує fuse. lr_thr=0 замість 0.1, щоб крива skydet ішла нижче 0.1:
# top-5 сортується за conf, тож для ts >= 0.1 результат той самий.
SCFG.update(model=f"{W}/skydet/adgf_lite.onnx", topk=5, lr_thr=0.0)
PROV = {"cuda_fp32": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "trt_fp16": [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                    "trt_engine_cache_path": f"{OUT}/trt_cache"}),
                     "CUDAExecutionProvider", "CPUExecutionProvider"]}
SKY = {k: SkyDetector(SCFG, providers=p) for k, p in PROV.items()}
for k, d in SKY.items(): print("skydet", k, "->", d.sess.get_providers(), flush=True)
assert SKY["trt_fp16"].provider == "TensorrtExecutionProvider" and SKY["cuda_fp32"].provider == "CUDAExecutionProvider"


# ---------- передбачення (кеш) ----------
PY = {k: cached(f"yolo_trt_fp16_{k}", lambda it=it: E.predict(ym, it, batch=1)) for k, it in SETS.items()}
PYPT = {k: cached(f"yolo_pt_fp32_{k}", lambda it=it: E.predict(ypt, it, batch=16)) for k, it in SETS.items()}
PS = {b: {k: cached(f"skydet_{b}_{k}", lambda it=it, b=b: F.sky_predict(SKY[b], it)) for k, it in SETS.items()} for b in SKY}
print("передбачення готові", flush=True)


def short(r): return {c: r[c] for c in ["frames", "targets", "recall", "fp_per_frame", "fp_max", "recall_by_size"]}


# ---------- 2. точність ----------
acc = {"config": CFG, "rule": "влучання: центр рамки в рамці цілі +-3 px; Recall для цілей 5-50 px; top-1",
       "yolo_backend": "TensorRT FP16", "results": {}, "prob_diff": {}}
for k, it in SETS.items():
    r = {"yolo_trt_fp16": short(F.score(it, PY[k], no(PY[k]), CFG)),
         "yolo_pytorch_fp32": short(F.score(it, PYPT[k], no(PYPT[k]), CFG))}
    for b in SKY:
        r[f"skydet_{b}"] = short(F.score(it, no(PY[k]), PS[b][k], CFG))                              # ts=0.3, top-1
        r[f"skydet_native_top3_{b}"] = short(E.score(it, PS[b][k], dict(conf=0.3, topk=3, margin=0)))  # skydet_config.json
        r[f"pipeline_skydet_{b}"] = short(F.score(it, PY[k], PS[b][k], CFG))
    acc["results"][k] = r
    # різниця карт імовірності FP32 / FP16 і кадри, де top-1 плями skydet розійшлись
    d, flips = [], 0
    for x in it[:: max(1, len(it) // 100)]:
        img = E.load(x)[:, :, 0]
        d.append(float(np.abs(SKY["cuda_fp32"].prob(img) - SKY["trt_fp16"].prob(img)).max()))
    for a, b in zip(PS["cuda_fp32"][k], PS["trt_fp16"][k]):
        a, b = a[a[:, 4] >= CFG["ts"]], b[b[:, 4] >= CFG["ts"]]
        flips += len(a) != len(b) or (len(a) and float(np.abs(a[:, :4] - b[:, :4]).max()) > 1)
    acc["prob_diff"][k] = dict(max_abs=round(max(d), 4), mean_of_max=round(float(np.mean(d)), 4), frames_checked=len(d),
                               frames_with_different_skydet_blobs=int(flips))
acc["recall_drop_pp"] = {}
for k in SETS:
    r = acc["results"][k]
    if r["skydet_cuda_fp32"]["recall"] is None: continue
    acc["recall_drop_pp"][k] = {s: round(100 * (r[f"{s}_cuda_fp32"]["recall"] - r[f"{s}_trt_fp16"]["recall"]), 2)
                                for s in ["skydet", "skydet_native_top3", "pipeline_skydet"]}
acc["trt_fp16_ok"] = all(v <= 0.5 for d in acc["recall_drop_pp"].values() for v in d.values())
json.dump(acc, open(f"{OUT}/accuracy.json", "w"), indent=1, ensure_ascii=False)
for k, r in acc["results"].items():
    print(k)
    for s, v in r.items(): print(f"  {s:34s} recall {v['recall']}  fp/кадр {v['fp_per_frame']}  макс {v['fp_max']}")
print("падіння Recall, п.п.:", acc["recall_drop_pp"], "ok:", acc["trt_fp16_ok"], flush=True)


# ---------- 3. криві ----------
rows = []
for b in SKY:
    for k in ["sirst_sky_targets", "sirst_no_targets"]:
        it, py, ps = SETS[k], PY[k], PS[b][k]
        def add(system, swept, t, cfg, y, s):
            r = F.score(it, y, s, cfg)
            rows.append(dict(system=system, set=k, skydet_backend=b, swept=swept, threshold=t, ty=cfg["ty"], ts=cfg["ts"],
                             a=cfg["a"], recall=r["recall"], fp_per_frame=r["fp_per_frame"], fp_max=r["fp_max"]))
        for t in THRS:
            if b == "cuda_fp32":                    # YOLO не залежить від бекенду skydet
                add("yolo", "ty", t, dict(ty=t, ts=1.1, a=CFG["a"]), py, no(ps))
            add("skydet", "ts", t, dict(ty=1.1, ts=t, a=CFG["a"]), no(py), ps)
            add("yolo+skydet", "ty", t, dict(CFG, ty=t), py, ps)      # ts=0.3 фіксований
            add("yolo+skydet", "ts", t, dict(CFG, ts=t), py, ps)      # ty=0.15 фіксований
for r in rows:
    if r["system"] == "yolo": r["skydet_backend"] = ""; r["ts"] = ""
    if r["system"] == "skydet": r["ty"] = ""
with open(f"{OUT}/curves.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("curves.csv", len(rows), "рядків", flush=True)


# ---------- 1. швидкість ----------
allit = [x for it in SETS.values() for x in it]
imgs = [E.load(x) for x in allit[:: max(1, len(allit) // 50)][:50]]
assert all(i.shape == (512, 640, 3) for i in imgs)


def stat(t):
    return dict(ms_mean=round(float(np.mean(t)), 2), ms_median=round(float(np.median(t)), 2),
                ms_p95=round(float(np.percentile(t, 95)), 2), fps=round(1000 / float(np.mean(t)), 1))


def bench(run, n=N, warm=50):
    for i in range(warm): run(imgs[i % len(imgs)])
    t = []
    for i in range(n):
        t0 = time.perf_counter(); run(imgs[i % len(imgs)]); t.append((time.perf_counter() - t0) * 1000)
    return stat(t)


def run_yolo(img):                                  # як E.speed: підготовка, модель, top-1
    bx = ym.predict(img, imgsz=[512, 640], conf=CFG["ty"], verbose=False)[0].boxes
    return E.keep(np.hstack([bx.xyxy.cpu().numpy(), bx.conf.cpu().numpy()[:, None]]), CFG["ty"], 1, 0)


def run_pipe(det):                                  # як F.speed: YOLO + skydet + злиття
    def f(img):
        bx = ym.predict(img, imgsz=[512, 640], conf=CFG["ty"], verbose=False)[0].boxes
        py = np.hstack([bx.xyxy.cpu().numpy(), bx.conf.cpu().numpy()[:, None]])
        return F.fuse(py, F.sky_pred(det, img[:, :, 0]), CFG["ty"], CFG["ts"], CFG["a"])
    return f


sp = {"yolo_trt_fp16": bench(run_yolo)}
for b, det in SKY.items():
    sp[f"skydet_{b}"] = bench(lambda img, det=det: F.sky_pred(det, img[:, :, 0]))            # нормалізація, модель, фільтри
    sp[f"skydet_{b}_model_only"] = bench(lambda img, det=det: det.prob(img[:, :, 0]))        # нормалізація + модель
for b, det in SKY.items():
    sp[f"pipeline_yolo_trt_fp16+skydet_{b}"] = bench(run_pipe(det))
# повтор у зворотному порядку: перевірка, що порядок замірів не впливає
rep = {f"pipeline_yolo_trt_fp16+skydet_{b}": bench(run_pipe(SKY[b])) for b in reversed(list(SKY))}
rep["yolo_trt_fp16"] = bench(run_yolo)
speed = dict(gpu=torch.cuda.get_device_name(0), frame="640x512", batch=1, frames=N, warmup=50,
             note="час на кадрі в пам'яті: підготовка, модель, постобробка, злиття; 50 кадрів SIRST (sky_targets, sun_real, "
                  "no_targets) по колу; без читання з диска",
             fusion=dict(CFG, topk=1), skydet_cfg=dict(topk=SCFG["topk"], lr_thr=SCFG["lr_thr"], thr=SCFG["thr"]),
             versions=dict(torch=torch.__version__, tensorrt=trt.__version__, onnxruntime=ort.__version__,
                           ultralytics=__import__("ultralytics").__version__),
             results=sp, repeat=rep)
json.dump(speed, open(f"{OUT}/speed.json", "w"), indent=1, ensure_ascii=False)
for k, v in sp.items(): print(f"{k:44s} {v}")
for k, v in rep.items(): print(f"повтор {k:37s} {v}")
