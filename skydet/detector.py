"""Детектор малих цілей на небі (кадр 640x512).

Кадр -> нормалізація -> ADGFNet-Lite (ONNX) -> поріг -> плями -> маска сонця -> розмір -> фільтр хибних плям -> top-K.
Для відео додатково часове підтвердження (TemporalFilter).
"""
import json, time
import numpy as np, cv2, onnxruntime as ort

H, W = 512, 640

DEFAULT = dict(
    model="adgf_lite.onnx",
    mean=87.4661865234375, std=39.71953201293945,   # нормалізація, з якою вчили ваги IRSTD-1k
    thr=0.5,              # поріг ймовірності
    min_side=3, max_side=60,   # фільтр розміру плями (ціль 5-50 px + запас)
    sun_level=250, sun_area=200, sun_margin=15,   # маска сонця
    topk=3,               # максимум детекцій на кадр, 0 = без обмеження
    bg_grad_max=0, bg_ring=12,   # простий фільтр фону за градієнтом (за замовчуванням вимкнений, його замінює lr)
    lr_coef=None, lr_thr=0.3,    # фільтр хибних плям: логістична регресія на 8 ознаках плями і фону, вчиться на val (evaluate.py --tune)
    use_sun=True, use_size=True, use_bg=False, use_lr=True, use_topk=True,
    backend="auto",       # auto | cpu | cuda | trt_fp16 (TensorRT FP16 через ONNX Runtime)
)


def to8bit(img):
    """16-бітний кадр -> 8 біт по перцентилях 0.5-99.5%. 8-бітний не чіпаємо."""
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, (0.5, 99.5))
    return np.clip((img.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)


def fit(img, pad=None):
    """Будь-який кадр -> 640x512: більший зменшуємо, менший доповнюємо медіаною кадру
    (чорні поля погіршують Recall: 92.1% проти 93.7% на IRSTD-1k). Повертає кадр, масштаб, розмір без поля."""
    h, w = img.shape
    s = min(1.0, W / w, H / h)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    out = np.full((H, W), int(np.median(img)) if pad is None else pad, np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    return out, s, img.shape


def bg_grad(img, x, y, w, h, ring):
    """Середній модуль градієнта в кільці ring px навколо плями (саму пляму з запасом 2 px виключаємо).
    Небо і хмари дають малі значення, будівлі, дерева, дроти, дорога дають великі."""
    X0, Y0, X1, Y1 = max(x - ring - 2, 0), max(y - ring - 2, 0), min(x + w + ring + 2, W), min(y + h + ring + 2, H)
    P = img[Y0:Y1, X0:X1].astype(np.float32)
    g = np.hypot(cv2.Sobel(P, cv2.CV_32F, 1, 0), cv2.Sobel(P, cv2.CV_32F, 0, 1))
    inner = np.zeros(P.shape, bool); inner[max(y - 2 - Y0, 0):y + h + 2 - Y0, max(x - 2 - X0, 0):x + w + 2 - X0] = True
    return float(g[~inner].mean()) if (~inner).any() else 0.0


def features(img, d):
    """Ознаки плями для фільтра: градієнт фону (кільця 4 і 12 px), контраст, контраст/шум, площа, score, яскравість фону."""
    x, y, w, h = d["bbox"]
    g4, g12 = bg_grad(img, x, y, w, h, 4), bg_grad(img, x, y, w, h, 12)
    X0, Y0, X1, Y1 = max(x - 8, 0), max(y - 8, 0), min(x + w + 8, W), min(y + h + 8, H)
    P = img[Y0:Y1, X0:X1].astype(np.float32)
    inner = np.zeros(P.shape, bool); inner[max(y - 2 - Y0, 0):y + h + 2 - Y0, max(x - 2 - X0, 0):x + w + 2 - X0] = True
    bg = P[~inner] if (~inner).any() else P.ravel()
    t = img[y:y + h, x:x + w].astype(np.float32); med = float(np.median(bg))
    c = max(t.max() - med, med - t.min())
    return np.array([np.log1p(g4), np.log1p(g12), np.log1p(c), np.log1p(c / (bg.std() + 1)), np.log1p(d["area_px"]),
                     d["score"], med / 255, np.log1p(g12) - np.log1p(c)], np.float32)


def sun_mask(img, level, area, margin):
    """Пересвічені області (сонце, відблиски) площею >= area, розширені на margin px."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((img >= level).astype(np.uint8), 8)
    big = np.isin(lab, [i for i in range(1, n) if st[i, 4] >= area])
    if not big.any():
        return None
    return cv2.dilate(big.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)) > 0


class SkyDetector:
    def __init__(self, cfg=None, providers=None):
        self.cfg = {**DEFAULT, **(cfg or {})}
        b = self.cfg["backend"]
        trt = ("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": "trt_cache"})
        providers = providers or {"cpu": ["CPUExecutionProvider"], "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                                  "trt_fp16": [trt, "CUDAExecutionProvider", "CPUExecutionProvider"]}.get(
            b, [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()])
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(self.cfg["model"], so, providers=providers)
        self.provider = self.sess.get_providers()[0]

    def prob(self, img8):
        x = ((img8.astype(np.float32) - self.cfg["mean"]) / self.cfg["std"])[None, None]
        return self.sess.run(None, {"img": x})[0][0, 0]

    def post(self, p, img8, valid_hw=(H, W)):
        c = self.cfg
        n, lab, st, cen = cv2.connectedComponentsWithStats((p > c["thr"]).astype(np.uint8), 8)
        sm = sun_mask(img8, c["sun_level"], c["sun_area"], c["sun_margin"]) if c["use_sun"] else None
        dets = []
        for i in range(1, n):
            x, y, w, h, a = (int(v) for v in st[i])
            if y >= valid_hw[0] or x >= valid_hw[1]:
                continue                                   # пляма на доповненій рамці
            if c["use_size"] and not (c["min_side"] <= max(w, h) <= c["max_side"]):
                continue
            cx, cy = float(cen[i, 0]), float(cen[i, 1])
            if sm is not None and sm[int(round(cy)), int(round(cx))]:
                continue
            if c["use_bg"] and c["bg_grad_max"] and bg_grad(img8, x, y, w, h, c["bg_ring"]) > c["bg_grad_max"]:
                continue
            d = dict(bbox=[x, y, w, h], center=[round(cx, 1), round(cy, 1)],
                     score=round(float(p[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].max()), 3), area_px=a)
            if c["use_lr"] and c["lr_coef"]:
                wb = np.asarray(c["lr_coef"], np.float32)
                d["conf"] = round(float(1 / (1 + np.exp(-(features(img8, d) @ wb[:-1] + wb[-1])))), 3)
                if d["conf"] < c["lr_thr"]:
                    continue
            dets.append(d)
        dets.sort(key=lambda d: -d.get("conf", d["score"]))
        if c["use_topk"] and c["topk"]:
            dets = dets[:c["topk"]]
        return dets, sm is not None

    def __call__(self, frame, frame_id=""):
        t0 = time.perf_counter()
        img, s, hw = fit(to8bit(frame))
        p = self.prob(img)
        dets, sun = self.post(p, img, hw)
        if s < 1:                                          # координати назад у вихідний кадр
            for d in dets:
                d["bbox"] = [round(v / s) for v in d["bbox"]]; d["center"] = [round(v / s, 1) for v in d["center"]]
        return dict(frame_id=str(frame_id), time_ms=round((time.perf_counter() - t0) * 1000, 2), sun_flag=sun, targets=dets)


class TemporalFilter:
    """Відео: ціль підтверджена, якщо поруч (radius px) була детекція хоча б у need з останніх window кадрів."""
    def __init__(self, window=3, need=2, radius=10):
        self.window, self.need, self.r, self.hist = window, need, radius, []

    def __call__(self, dets):
        self.hist = (self.hist + [[d["center"] for d in dets]])[-self.window:]
        ok = []
        for d in dets:
            hits = sum(any(abs(d["center"][0] - c[0]) <= self.r and abs(d["center"][1] - c[1]) <= self.r for c in fr) for fr in self.hist)
            if hits >= self.need:
                ok.append(d)
        return ok


def load_cfg(path):
    return {**DEFAULT, **json.load(open(path))} if path else dict(DEFAULT)
