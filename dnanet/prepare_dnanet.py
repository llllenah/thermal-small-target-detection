"""Build a DNA-Net style dataset (YeRen123455/Infrared-Small-Target-Detection) from
HIT-UAV, FLIR driving frames and SIRST-v2.

Output layout (what DNA-Net's load_dataset / TrainSetLoader expect):
  <out>/THERMAL-MIX/images/<id>.png   8-bit grayscale
  <out>/THERMAL-MIX/masks/<id>.png    0 / 255
  <out>/THERMAL-MIX/ours/train.txt    ids for training
  <out>/THERMAL-MIX/ours/test.txt     = validation ids (DNA-Net picks its best epoch on test.txt)
  <out>/THERMAL-MIX/ours/final_test.txt + final_test_<subset>.txt   held-out, touch once at the end
  <out>/THERMAL-MIX/meta.json          per-image: source, group, split, targets (boxes, sizes)

Rules (see CLAUDE.md):
  * frames larger than 640x512 are downscaled (aspect kept), smaller ones are kept as is;
  * boxes from HIT-UAV / FLIR become pseudo-masks: pixels inside the box that differ from the
    local background (warm OR cold), clipped to an ellipse; fallback = the ellipse itself;
  * splits are made by video group, never by frame (the official HIT-UAV split leaks);
  * optional synthetic sun: a saturated disc with a glow pasted away from targets.
"""
import os, re, json, glob, random, argparse, hashlib
import xml.etree.ElementTree as ET
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- loaders
def load_hit(root):
    out = []
    for split in ["train", "val", "test"]:
        d = json.load(open(f"{root}/normal_json/annotations/{split}.json"))
        by = {}
        for a in d["annotation"]:
            if a["category_id"] == 4:  # DontCare
                continue
            by.setdefault(a["image_id"], []).append(a["bbox"])
        for im in d["images"]:
            fn = im["filename"]
            dn, alt, ang, wx, idx = fn[:-4].split("_")
            out.append(dict(src="hit", name=fn[:-4], path=f"{root}/normal_json/{split}/{fn}",
                            boxes=[tuple(map(float, b)) for b in by.get(im["id"], [])], mask=None,
                            group=f"hit_{dn}_{alt}_{ang}_{wx}", night=dn == "1"))
    return out


def load_flir(root):
    out = []
    for x in sorted(glob.glob(f"{root}/driving_annotation/*.xml")):
        r = ET.parse(x).getroot()
        fn = r.find("filename").text
        bx = []
        for o in r.findall("object"):
            b = o.find("bndbox")
            x1, y1, x2, y2 = [float(b.find(k).text) for k in ["xmin", "ymin", "xmax", "ymax"]]
            bx.append((x1, y1, x2 - x1, y2 - y1))
        frame = int(re.findall(r"_(\d+)\.png$", fn)[0])
        # one long drive: consecutive blocks of 2000 frame numbers form a "group"
        out.append(dict(src="flir", name=fn[:-4], path=f"{root}/driving_data/{fn}", boxes=bx, mask=None,
                        group=f"flir_{frame // 2000}", night=False))
    return out


def load_sirst(root):
    out = []
    for p in sorted(glob.glob(f"{root}/mixed/*.png")):
        n = os.path.basename(p)[:-4]
        m = cv2.imread(f"{root}/annotations/masks/{n}_pixels0.png", 0)
        g = re.sub(r"[-_]\d+$", "", n) + ("_bg" if not (m > 127).any() else "")          # S20210713_S3_3 -> S20210713_S3 (one sequence)
        if n.startswith("Misc_"):
            g = n                                # SIRST-v1 images are independent shots
        out.append(dict(src="sirst", name=n, path=p, boxes=None, mask=m, group=f"sirst_{g}", night=False))
    return out


def load_irstd(root):
    """IRSTD-1k with masks. Accepts the original release (IRSTD1k_Img/, IRSTD1k_Label/)
    or the BasicIRSTD layout (images/, masks/)."""
    img_dir = next((f"{root}/{d}" for d in ["IRSTD1k_Img", "images"] if os.path.isdir(f"{root}/{d}")), None)
    msk_dir = next((f"{root}/{d}" for d in ["IRSTD1k_Label", "masks"] if os.path.isdir(f"{root}/{d}")), None)
    out = []
    if img_dir is None or msk_dir is None:
        print("IRSTD-1k: images/masks folders not found in", root)
        return out
    for p in sorted(glob.glob(f"{img_dir}/*.png")):
        n = os.path.basename(p)[:-4]
        m = cv2.imread(f"{msk_dir}/{n}.png", 0)
        if m is None:
            continue
        out.append(dict(src="irstd", name=n, path=p, boxes=None, mask=m, group=f"irstd_{n}", night=False))
    return out


def load_antiuav(root, step=10):
    """Anti-UAV410: <split>/<sequence>/*.jpg + IR_label.json {"exist": [...], "gt_rect": [[x,y,w,h], ...]}.
    Every `step`-th frame is taken; frames where the drone is absent stay as target-free frames."""
    out = []
    for lab in sorted(glob.glob(f"{root}/*/*/IR_label.json")):
        seq_dir = os.path.dirname(lab)
        seq = os.path.basename(seq_dir)
        js = json.load(open(lab))
        frames = sorted(glob.glob(f"{seq_dir}/*.jpg"))
        exist, rect = js.get("exist", []), js.get("gt_rect", [])
        for k in range(0, min(len(frames), len(rect)), step):
            r = rect[k] if k < len(rect) else []
            boxes = [tuple(map(float, r))] if (k < len(exist) and exist[k] and len(r) == 4 and r[2] > 0 and r[3] > 0) else []
            out.append(dict(src="antiuav", name=f"{seq}_{k:05d}", path=frames[k], boxes=boxes, mask=None,
                            group=f"antiuav_{seq}", night=False))
    return out


# ---------------------------------------------------------------- image ops
def gray(p):
    im = cv2.imread(p, -1)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return im


def fit640(im, mask=None, boxes=None):
    h, w = im.shape
    s = min(1.0, 640 / w, 512 / h)
    if s < 1.0:
        size = (round(w * s), round(h * s))
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        if mask is not None:  # area-resize then low threshold keeps 1-2 px targets alive
            mask = (cv2.resize((mask > 127).astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.25).astype(np.uint8) * 255
        if boxes is not None:
            boxes = [(x * s, y * s, bw * s, bh * s) for x, y, bw, bh in boxes]
    return im, mask, boxes


def pseudo_mask(im, boxes):
    """Box -> pixel mask. Pixels in the box whose |I - local background| passes Otsu, kept inside
    a slightly enlarged inscribed ellipse; largest blob only. Falls back to the ellipse."""
    H, W = im.shape
    m = np.zeros((H, W), np.uint8)
    stats = dict(otsu=0, ellipse=0)
    for x, y, w, h in boxes:
        x0, y0 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
        x1, y1 = min(int(np.ceil(x + w)), W), min(int(np.ceil(y + h)), H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        bw, bh = x1 - x0, y1 - y0
        ell = np.zeros((bh, bw), np.uint8)
        cv2.ellipse(ell, (bw // 2, bh // 2), (max(1, int(bw * 0.575)), max(1, int(bh * 0.575))), 0, 0, 360, 1, -1)
        p = max(bw, bh) // 2 + 2
        X0, Y0, X1, Y1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
        ring = im[Y0:Y1, X0:X1].astype(np.float32)
        rm = np.ones_like(ring, bool); rm[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        bg = np.median(ring[rm]) if rm.any() else np.median(ring)
        diff = np.abs(im[y0:y1, x0:x1].astype(np.float32) - bg)
        sel = None
        if diff.max() >= 4:
            d8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, th = cv2.threshold(d8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th = th & ell
            n, lab, st, _ = cv2.connectedComponentsWithStats(th, 8)
            if n > 1:
                k = 1 + int(np.argmax(st[1:, 4]))
                blob = (lab == k).astype(np.uint8)
                blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                frac = blob.sum() / ell.sum()
                if 0.2 <= frac <= 1.0:
                    sel = blob
        if sel is None:
            sel = ell; stats["ellipse"] += 1
        else:
            stats["otsu"] += 1
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], sel * 255)
    return m, stats


def targets_from_mask(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    return [dict(x=int(st[i, 0]), y=int(st[i, 1]), w=int(st[i, 2]), h=int(st[i, 3]), area=int(st[i, 4])) for i in range(1, n)]


def add_sun(im, mask, rng):
    """Paste a saturated disc with a soft glow where it does not touch any target."""
    H, W = im.shape
    r = rng.randint(10, 35)
    dil = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3 * r, 3 * r), np.uint8))
    for _ in range(50):
        cx, cy = rng.randint(r, W - r - 1), rng.randint(r, H // 2)  # sun is usually in the upper half
        if not dil[cy, cx]:
            break
    else:
        return None
    yy, xx = np.mgrid[:H, :W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = np.clip(1.0 - (d - r) / (r * rng.uniform(2.0, 4.0)), 0, 1) ** 2
    out = im.astype(np.float32)
    out = out + (255 - out) * glow * rng.uniform(0.5, 0.9)
    out[d <= r] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- splits
def split_groups(items, seed, test_frac=0.15, val_frac=0.15):
    """Assign whole groups to train / val / final_test, per source, aiming at the frame fractions."""
    assign = {}
    key = lambda i: i["src"] + ("_bg" if i["src"] == "sirst" and not (i["mask"] > 127).any() else "")
    for src in sorted(set(key(i) for i in items)):
        cnt = {}
        for i in items:
            if key(i) == src:
                cnt[i["group"]] = cnt.get(i["group"], 0) + 1
        gs = sorted(cnt)
        random.Random(f"{seed}-{src}").shuffle(gs)  # per-source seed: adding a new dataset never reshuffles old splits
        total = sum(cnt.values()); acc = 0
        for g in gs:
            f = acc / total
            assign[g] = "final_test" if f < test_frac else ("val" if f < test_frac + val_frac else "train")
            acc += cnt[g]
    return assign


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit", default=f"{HERE}/../hituav")
    ap.add_argument("--flir", default=f"{HERE}/../probe/flirx")
    ap.add_argument("--sirst", default=f"{HERE}/../sirstv2")
    ap.add_argument("--irstd", default="", help="IRSTD-1k folder (optional, second wave)")
    ap.add_argument("--antiuav", default="", help="Anti-UAV410 folder with train/val/test (optional, second wave)")
    ap.add_argument("--antiuav_step", type=int, default=10, help="take every N-th video frame")
    ap.add_argument("--out", default=f"{HERE}/../dnanet_data")
    ap.add_argument("--name", default="THERMAL-MIX")
    ap.add_argument("--sun_train", type=float, default=0.10, help="extra synthetic-sun copies, share of train frames")
    ap.add_argument("--sun_test", type=int, default=150, help="synthetic-sun frames made from final_test frames")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = f"{a.out}/{a.name}"
    for d in ["images", "masks", "ours"]:
        os.makedirs(f"{root}/{d}", exist_ok=True)

    items = []
    if os.path.isdir(a.hit): items += load_hit(a.hit)
    if os.path.isdir(a.flir): items += load_flir(a.flir)
    if os.path.isdir(a.sirst): items += load_sirst(a.sirst)
    if a.irstd and os.path.isdir(a.irstd): items += load_irstd(a.irstd)
    if a.antiuav and os.path.isdir(a.antiuav): items += load_antiuav(a.antiuav, a.antiuav_step)
    print({s: sum(i["src"] == s for i in items) for s in sorted(set(i["src"] for i in items))})
    assign = split_groups(items, a.seed)
    rng = random.Random(a.seed)

    meta, lists, pm = {}, {"train": [], "val": [], "final_test": []}, dict(otsu=0, ellipse=0)
    final_sub = {}
    for it in items:
        im = gray(it["path"])
        if it["mask"] is not None:
            im, mask, _ = fit640(im, mask=it["mask"])
        else:
            im, _, boxes = fit640(im, boxes=it["boxes"])
            mask, st = pseudo_mask(im, boxes)
            pm["otsu"] += st["otsu"]; pm["ellipse"] += st["ellipse"]
        iid = f"{it['src']}_{it['name']}"
        cv2.imwrite(f"{root}/images/{iid}.png", im)
        cv2.imwrite(f"{root}/masks/{iid}.png", mask)
        sp = assign[it["group"]]
        lists[sp].append(iid)
        tg = targets_from_mask(mask)
        meta[iid] = dict(src=it["src"], group=it["group"], split=sp, night=it["night"], H=im.shape[0], W=im.shape[1],
                         targets=tg, synthetic_sun=False)
        if sp == "final_test":
            sub = "sirst_background" if (it["src"] == "sirst" and not tg) else it["src"]
            final_sub.setdefault(sub, []).append(iid)

    # synthetic sun: extra train copies + a held-out sun test subset
    n_train_sun = int(len(lists["train"]) * a.sun_train)
    for sp, pool, n in [("train", [i for i in lists["train"] if not i.startswith("sirst")], n_train_sun),
                        ("final_test", [i for i in lists["final_test"] if not i.startswith("sirst")], a.sun_test)]:
        rng.shuffle(pool)
        made = 0
        for iid in pool:
            if made >= n:
                break
            im = cv2.imread(f"{root}/images/{iid}.png", 0); mask = cv2.imread(f"{root}/masks/{iid}.png", 0)
            s = add_sun(im, mask, rng)
            if s is None:
                continue
            nid = f"sun_{iid}"
            cv2.imwrite(f"{root}/images/{nid}.png", s); cv2.imwrite(f"{root}/masks/{nid}.png", mask)
            lists[sp].append(nid)
            meta[nid] = dict(meta[iid], synthetic_sun=True, split=sp)
            if sp == "final_test":
                final_sub.setdefault("sun", []).append(nid)
            made += 1

    w = lambda f, ids: open(f"{root}/ours/{f}", "w").write("\n".join(ids) + "\n")
    w("train.txt", lists["train"]); w("test.txt", lists["val"]); w("final_test.txt", lists["final_test"])
    for k, v in final_sub.items():
        w(f"final_test_{k}.txt", v)
    json.dump(meta, open(f"{root}/meta.json", "w"))

    # BasicIRSTD layout (github.com/XinyiYing/BasicIRSTD): <name>/img_idx/{train,test}_<name>.txt,
    # plus one sibling "dataset" per held-out subset that reuses the same images/masks via symlinks,
    # so test.py can score each subset separately: --dataset_names <name>-FT-hit <name>-FT-sun ...
    os.makedirs(f"{root}/img_idx", exist_ok=True)
    open(f"{root}/img_idx/train_{a.name}.txt", "w").write("\n".join(lists["train"]) + "\n")
    open(f"{root}/img_idx/test_{a.name}.txt", "w").write("\n".join(lists["val"]) + "\n")
    for k, v in list(final_sub.items()) + [("all", lists["final_test"])]:
        sub = f"{a.name}-FT-{k}"
        sroot = f"{a.out}/{sub}"
        os.makedirs(f"{sroot}/img_idx", exist_ok=True)
        for d in ["images", "masks"]:
            if not os.path.lexists(f"{sroot}/{d}"):
                os.symlink(os.path.abspath(f"{root}/{d}"), f"{sroot}/{d}")
        open(f"{sroot}/img_idx/test_{sub}.txt", "w").write("\n".join(v) + "\n")

    # summary
    def summ(ids):
        t = [x for i in ids for x in meta[i]["targets"]]
        s = [np.sqrt(x["w"] * x["h"]) for x in t]
        return dict(frames=len(ids), targets=len(t), empty_frames=sum(not meta[i]["targets"] for i in ids),
                    in_5_50=round(float(np.mean([(min(x["w"], x["h"]) >= 5) and (max(x["w"], x["h"]) <= 50) for x in t])) * 100, 1) if t else 0)
    rep = {k: summ(v) for k, v in [("train", lists["train"]), ("val(test.txt)", lists["val"]), ("final_test", lists["final_test"])]}
    rep.update({f"final_test_{k}": summ(v) for k, v in final_sub.items()})
    rep["pseudo_masks"] = pm
    rep["groups"] = {s: len([g for g, v in assign.items() if v == s]) for s in ["train", "val", "final_test"]}
    json.dump(rep, open(f"{root}/summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
