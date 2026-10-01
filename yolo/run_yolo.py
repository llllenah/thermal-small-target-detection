"""YOLO для кейсу 4.5: швидкий перебір моделей, повне навчання, TensorRT FP16/INT8, оцінка за ТЗ.

  python run_yolo.py sweep  --data auav_sweep/data.yaml                     # кроки 2-3: яка модель і голова (P3 чи P2)
                            # auav_sweep = prepare_yolo.py --stride 30 (у 3 рази менше кадрів, ті самі серії)
  python run_yolo.py train  --data auav_yolo/data.yaml --model yolo26l-p2   # крок 5
  python run_yolo.py export --weights runs/.../best.pt --data auav_yolo/data.yaml   # кроки 6-7: engine FP16 і INT8
  python run_yolo.py eval   --weights best.engine --val auav_yolo:val --test auav_yolo:test sky=testsets/sky_targets ...

Оцінка така сама, як у skydet: влучання = центр рамки в рамці цілі +-3 px, цілі <5 і >50 px ігноруються,
FP рахується на кожному наборі окремо, поріг conf підбирається на val (FP <= fp_budget), FPS = увесь predict на кадрі 640x512.
"""
import os, sys, json, time, glob, argparse
import numpy as np, cv2

TOL = 3


def gts_yolo(lbl, W=640, H=512):
    out = []
    if os.path.exists(lbl):
        for l in open(lbl).read().split("\n"):
            if l.strip():
                _, cx, cy, w, h = map(float, l.split()); w, h = w * W, h * H
                out.append((cx * W - w / 2, cy * H - h / 2, w, h))
    return out


def gts_mask(m):
    n, _, st, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
    return [tuple(map(float, st[i, :4])) for i in range(1, n)]


def load_set(spec):
    """'dir:split' = YOLO-датасет (images/split, labels/split); 'name=dir' = набір skydet (images/, masks/, list.txt)."""
    if "=" in spec:
        name, d = spec.split("=", 1)
        items = [(f"{d}/images/{n}.png", gts_mask(cv2.imread(f"{d}/masks/{n}.png", 0))) for n in open(f"{d}/list.txt").read().split()]
    else:
        d, split = spec.rsplit(":", 1); name = split
        items = [(p, gts_yolo(f"{d}/labels/{split}/{os.path.basename(p).rsplit('.', 1)[0]}.txt"))
                 for p in sorted(glob.glob(f"{d}/images/{split}/*"))]
    return name, items


def predict_all(model, items, imgsz):
    out = []
    for p, g in items:
        r = model.predict(cv2.imread(p), imgsz=imgsz, conf=0.01, verbose=False)[0].boxes
        out.append((r.xywh.cpu().numpy(), r.conf.cpu().numpy(), g))
    return out


def score(preds, conf, topk):
    tp = n_ok = 0; fps = []
    for xywh, c, gts in preds:
        keep = np.argsort(-c)[: topk or None]; keep = [i for i in keep if c[i] >= conf]
        hit, fp = set(), 0
        for i in keep:
            cx, cy = xywh[i, 0], xywh[i, 1]
            h = [j for j, (x, y, w, hh) in enumerate(gts) if x - TOL <= cx <= x + w + TOL and y - TOL <= cy <= y + hh + TOL]
            fp += not h
            hit.update(j for j in h if min(gts[j][2], gts[j][3]) >= 5 and max(gts[j][2], gts[j][3]) <= 50)
        tp += len(hit); n_ok += sum(min(w, h) >= 5 and max(w, h) <= 50 for _, _, w, h in gts); fps.append(fp)
    fps = np.array(fps)
    return dict(frames=len(preds), targets=int(n_ok), recall=round(tp / n_ok, 3) if n_ok else None,
                fp_per_frame=round(float(fps.mean()), 3), fp_max=int(fps.max()), frames_fp_gt1=round(float((fps > 1).mean()), 3))


def speed(model, imgsz, n=200):
    img = np.full((512, 640, 3), 90, np.uint8); cv2.circle(img, (300, 200), 4, (220,) * 3, -1)
    for _ in range(20): model.predict(img, imgsz=imgsz, verbose=False)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); model.predict(img, imgsz=imgsz, verbose=False); t.append((time.perf_counter() - t0) * 1000)
    return dict(ms_mean=round(float(np.mean(t)), 2), ms_p95=round(float(np.percentile(t, 95)), 2), fps=round(1000 / float(np.mean(t)), 1))


AUG = dict(hsv_h=0, hsv_s=0, hsv_v=0.3, degrees=0, scale=0.3, mosaic=1.0, close_mosaic=10, fliplr=0.5, mixup=0)


def build(name):
    """Повертає (модель, ваги для старту). P2-голова не має готових ваг: будуємо з yaml і стартуємо
    з ваг COCO звичайної моделі того ж розміру (параметр pretrained у train, працює і з 2 GPU)."""
    from ultralytics import YOLO
    if name.endswith(".pt") or name.endswith(".engine"):
        return YOLO(name), True
    if "-p2" in name:
        return YOLO(name + ".yaml"), name.replace("-p2", "") + ".pt"
    return YOLO(name + ".pt"), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "train", "export", "eval"])
    ap.add_argument("--data"); ap.add_argument("--model", default="yolo26l-p2"); ap.add_argument("--weights")
    ap.add_argument("--models", default="yolo26l,yolo26s-p2,yolo26m-p2,yolo26l-p2,yolo26x-p2")
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--val", nargs="*", default=[]); ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--fp_budget", type=float, default=0.7); ap.add_argument("--json", default="yolo_results.json")
    a = ap.parse_args()
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    if a.cmd == "sweep":                  # коротке навчання на частині даних, щоб порівняти моделі між собою
        rows = []
        for name in a.models.split(","):
            m, pre = build(name)
            m.train(data=a.data, pretrained=pre, epochs=15, imgsz=a.imgsz, batch=a.batch, project="sweep", name=name,
                    exist_ok=True, verbose=False, plots=False, **AUG)
            v = m.val(data=a.data, split="val", imgsz=a.imgsz, verbose=False)
            best, _ = build(f"sweep/{name}/weights/best.pt")
            rows.append(dict(model=name, params_M=round(sum(p.numel() for p in best.model.parameters()) / 1e6, 1),
                             mAP50=round(float(v.box.map50), 3), recall=round(float(v.box.mr), 3), gpu=gpu,
                             **{f"pytorch_{k}": x for k, x in speed(best, [512, 640]).items()}))
            print(rows[-1], flush=True); json.dump(rows, open("sweep.json", "w"), indent=1)

    elif a.cmd == "train":
        m, pre = build(a.model)
        m.train(data=a.data, pretrained=pre, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, fraction=a.fraction, project="train", name=a.model,
                exist_ok=True, patience=20, cos_lr=True, **AUG)

    elif a.cmd == "export":               # крок 6: TensorRT FP16; крок 7: INT8 з калібруванням на train
        from ultralytics import YOLO
        for kw in [dict(half=True), dict(int8=True, data=a.data, fraction=0.1)]:
            p = YOLO(a.weights).export(format="engine", imgsz=[512, 640], batch=1, **kw)
            os.replace(p, p.replace(".engine", "_int8.engine" if "int8" in kw else "_fp16.engine")); print("saved", p)

    elif a.cmd == "eval":
        from ultralytics import YOLO
        m = YOLO(a.weights, task="detect")
        val = [(n, predict_all(m, it, [512, 640])) for n, it in map(load_set, a.val)]
        best = None
        for conf in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
            for k in [1, 3, 5]:
                r = {n: score(p, conf, k) for n, p in val}
                ok = all(x["fp_per_frame"] <= a.fp_budget for x in r.values())
                rec = np.mean([x["recall"] for x in r.values() if x["recall"] is not None])
                if ok and (best is None or rec > best[0]): best = (rec, conf, k, r)
        if best is None:                  # жоден поріг не вклався в бюджет FP: беремо найсуворіший
            best = (0, 0.6, 1, {n: score(p, 0.6, 1) for n, p in val}); print("УВАГА: FP > fp_budget навіть при conf 0.6, top-1")
        _, conf, k, rv = best
        res = dict(weights=a.weights, gpu=gpu, conf=conf, topk=k, val=rv,
                   test={n: score(predict_all(m, it, [512, 640]), conf, k) for n, it in map(load_set, a.test)}, speed=speed(m, [512, 640]))
        print(json.dumps(res, indent=1, ensure_ascii=False)); json.dump(res, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
