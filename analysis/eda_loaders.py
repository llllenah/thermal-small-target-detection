import os, glob, json, re
import xml.etree.ElementTree as ET
import numpy as np, cv2

S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "eda_out"); os.makedirs(OUT, exist_ok=True)

def load_hit():
    items = []
    for split in ["train", "val", "test"]:
        d = json.load(open(f"{S}/hituav/normal_json/annotations/{split}.json"))
        boxes = {}
        for a in d["annotation"]:
            if a["category_id"] == 4:  # DontCare
                continue
            boxes.setdefault(a["image_id"], []).append((a["bbox"], a["category_id"]))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        for im in d["images"]:
            fn = im["filename"]
            sc, alt, ang, wx, _ = fn[:-4].split("_")
            items.append(dict(ds="HIT-UAV", split=split, path=f"{S}/hituav/normal_json/{split}/{fn}",
                              boxes=[(x, y, w, h, cats[c]) for (x, y, w, h), c in boxes.get(im["id"], [])],
                              meta=dict(night=sc == "1", alt=int(alt), angle=int(ang), rain=wx == "1")))
    return items

def load_flir():
    items = []
    for x in sorted(glob.glob(f"{S}/probe/flirx/driving_annotation/*.xml")):
        r = ET.parse(x).getroot()
        fn = r.find("filename").text
        bx = []
        for o in r.findall("object"):
            b = o.find("bndbox")
            x1, y1, x2, y2 = [int(float(b.find(k).text)) for k in ["xmin", "ymin", "xmax", "ymax"]]
            bx.append((x1, y1, x2 - x1, y2 - y1, o.find("name").text))
        items.append(dict(ds="FLIR (авто)", split="all", path=f"{S}/probe/flirx/driving_data/{fn}", boxes=bx, meta={}))
    return items

def load_sirst():
    items = []
    for p in sorted(glob.glob(f"{S}/sirstv2/mixed/*.png")):
        name = os.path.basename(p)
        m = cv2.imread(f"{S}/sirstv2/annotations/masks/{name[:-4]}_pixels0.png", 0)
        if m is None:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
        bx = [(int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]), "target") for i in range(1, n)]
        items.append(dict(ds="SIRST-v2", split="all", path=p, boxes=bx, meta=dict(v1=name.startswith("Misc_"))))
    return items

def gray(p):
    im = cv2.imread(p, -1)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return im

def to640(im, boxes):
    h, w = im.shape
    s = min(1.0, 640 / w, 512 / h)  # only downscale; smaller frames are kept as-is
    if abs(s - 1) < 1e-6:
        return im, boxes, 1.0
    im2 = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return im2, [(x * s, y * s, bw * s, bh * s, c) for x, y, bw, bh, c in boxes], s

def box_stats(im, b):
    x, y, w, h, c = b
    H, W = im.shape
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(max(x1, x0 + 1), W), min(max(y1, y0 + 1), H)
    t = im[y0:y1, x0:x1].astype(np.float32)
    pad = int(max(w, h, 3))
    X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    ring = im[Y0:Y1, X0:X1].astype(np.float32).copy()
    msk = np.ones_like(ring, bool); msk[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    r = ring[msk]
    if r.size < 10 or t.size == 0:
        return None
    tm = float(t.mean())
    scr = (tm - r.mean()) / (r.std() + 1.0)
    brighter = float((im >= tm).mean())
    return dict(scr=float(scr), dI=float(tm - r.mean()), brighter=brighter, polarity=1 if tm >= r.mean() else -1)

def frame_stats(im):
    sat = (im >= 250).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sat, 8)
    big = [st[i, 4] for i in range(1, n) if st[i, 4] >= 200]
    return dict(sat_frac=float(sat.mean()), big_sat=len(big) > 0, mean=float(im.mean()), p99=float(np.percentile(im, 99)))

