"""Anti-UAV410 (+ відібрані негативи) -> YOLO-датасет і ті самі кадри для DNA-Net (маски з рамок).

  python prepare_yolo.py --root Anti-UAV-410 --neg negatives --out auav_yolo --stride 10 --neg_frac 0.15 --dnanet

Anti-UAV410: train/val/test/<серія>/*.jpg + IR_label.json {"exist": [...], "gt_rect": [[x, y, w, h], ...]}.
- Спліт офіційний, по серіях: серії не перетинаються між train, val і test.
- Сусідні кадри майже однакові, тому беремо кожен stride-й кадр із ціллю.
- Негативи: кадри без цілі з папки --neg/<split>, їх частка neg_frac від усіх кадрів спліту.
- DNA-Net вчиться на тих самих кадрах: маска = пікселі в рамці, що відрізняються від фону (Otsu), інакше еліпс.
"""
import os, glob, json, argparse, random
import numpy as np, cv2


def clip_box(box, W, H):
    """Рамку, що виходить за край кадру, обрізаємо по кадру. None, якщо від неї лишилось менше 1 px."""
    x, y, w, h = box
    x0, y0, x1, y1 = max(x, 0.0), max(y, 0.0), min(x + w, W), min(y + h, H)
    return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 >= 1 and y1 - y0 >= 1 else None


def box_to_mask(img, boxes):
    H, W = img.shape
    m = np.zeros(img.shape, np.uint8)
    for b in boxes:
        b = clip_box(b, W, H)
        if b is None: continue
        x, y, w, h = b
        x0, y0 = int(x), int(y); x1, y1 = min(int(np.ceil(x + w)), W), min(int(np.ceil(y + h)), H)
        if x1 - x0 < 2 or y1 - y0 < 2: continue
        ell = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.ellipse(ell, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 1, -1)
        p = int(max(w, h) // 2 + 2)
        bg = np.median(img[max(y0 - p, 0):y1 + p, max(x0 - p, 0):x1 + p])
        diff = np.abs(img[y0:y1, x0:x1].astype(np.float32) - bg)
        _, th = cv2.threshold(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sel = th & ell
        if diff.max() < 4 or sel.sum() < 0.2 * ell.sum(): sel = ell
        m[y0:y1, x0:x1] |= sel * 255
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--neg"); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10); ap.add_argument("--neg_frac", type=float, default=0.15)
    ap.add_argument("--dnanet", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed); stats = {}
    for split in ["train", "val", "test"]:
        for d in ["images", "labels"] + (["masks"] if a.dnanet else []):
            os.makedirs(f"{a.out}/{d}/{split}", exist_ok=True)
        names, sizes = [], []
        for sd in sorted(glob.glob(f"{a.root}/{split}/*/")):
            seq = os.path.basename(sd.rstrip("/"))
            lf = next((f for f in ["IR_label.json", "infrared.json"] if os.path.exists(sd + f)), None)
            if lf is None: continue
            lab = json.load(open(sd + lf)); frames = sorted(glob.glob(sd + "*.jpg"))
            for i in range(0, min(len(frames), len(lab["exist"])), a.stride):
                box = lab["gt_rect"][i]
                if not lab["exist"][i] or len(box) != 4 or box[2] < 1 or box[3] < 1: continue
                img = cv2.imread(frames[i], 0); H, W = img.shape
                box = clip_box(box, W, H)                       # частина рамок Anti-UAV410 виходить за край кадру
                if box is None: continue
                x, y, w, h = box; n = f"{seq}_{i:06d}"
                cv2.imwrite(f"{a.out}/images/{split}/{n}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                open(f"{a.out}/labels/{split}/{n}.txt", "w").write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
                if a.dnanet: cv2.imwrite(f"{a.out}/masks/{split}/{n}.png", box_to_mask(img, [box]))
                names.append(n); sizes.append(max(w, h))
        n_pos = len(names)
        negs = sorted(glob.glob(f"{a.neg}/{split}/*.jpg")) if a.neg else []
        k = min(len(negs), round(a.neg_frac / (1 - a.neg_frac) * n_pos)); rng.shuffle(negs)
        for p in negs[:k]:
            n = "neg_" + os.path.basename(p)[:-4]; img = cv2.imread(p, 0)
            cv2.imwrite(f"{a.out}/images/{split}/{n}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(f"{a.out}/labels/{split}/{n}.txt", "w").close()
            if a.dnanet: cv2.imwrite(f"{a.out}/masks/{split}/{n}.png", np.zeros_like(img))
            names.append(n)
        s = np.array(sizes)
        stats[split] = dict(positives=n_pos, negatives=k, neg_share=round(k / max(len(names), 1), 3),
                            box_lt5=round(float((s < 5).mean()), 3) if len(s) else None,
                            box_5_50=round(float(((s >= 5) & (s <= 50)).mean()), 3) if len(s) else None,
                            box_gt50=round(float((s > 50).mean()), 3) if len(s) else None)
        if a.dnanet:
            os.makedirs(f"{a.out}/img_idx", exist_ok=True)
            open(f"{a.out}/img_idx/{split}.txt", "w").write("\n".join(f"{split}/{n}" for n in names))
        print(split, stats[split], flush=True)
    open(f"{a.out}/data.yaml", "w").write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: target\n")
    json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
