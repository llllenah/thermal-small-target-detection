"""Тестові набори (кадри <= 640x512, маски цілей):

  val_irstd      IRSTD-1k test, 201 кадр            -> підбір порогу
  val_sun        ті самі кадри + синтетичне сонце   -> підбір порогу
  sky_targets    SIRST-v2, кадри з цілями           -> Recall, FP (модель їх не бачила)
  val_bg         SIRST-v2 без цілей, половина серій -> підбір порогу фільтра фону
  no_targets     SIRST-v2 без цілей, інша половина  -> FP на хмарах, будівлях, ліхтарях (серії не перетинаються з val_bg)
  sun_synth      SIRST-v2 + синтетичне сонце        -> FP на сонці, ціль поруч зі сонцем
  sun_real       SIRST-v2 кадри з реальним пересвіченням >= 200 px (яскраві хмари, дим, нагріті поверхні)

  python make_testsets.py --irstd ../../work/irstd --adgf ../../work/ADGFNet/ADGFNet --sirst ../../work/sirstv2 --out testsets
"""
import os, glob, argparse, random
import numpy as np, cv2


def shrink(img, m):
    h, w = img.shape; s = min(1.0, 640 / w, 512 / h)
    if s < 1:
        size = (round(w * s), round(h * s))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        m = ((cv2.resize((m > 127).astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.25) * 255).astype(np.uint8)
    return img, m


def add_sun(img, mask, rng):
    """Сонце: пересвічений диск радіусом 8-30 px з ореолом. Ставимо не ближче 40 px до цілей,
    у половині випадків відносно близько (40-120 px), щоб перевірити, чи не губиться ціль поруч."""
    h, w = img.shape
    ys, xs = np.nonzero(mask)
    r = rng.uniform(8, 30); sig = rng.uniform(1.0, 3.0) * r
    for _ in range(200):
        if len(xs) and rng.random() < 0.5:
            i = rng.integers(len(xs)); ang = rng.uniform(0, 2 * np.pi); d = rng.uniform(40, 120) + r
            cx, cy = xs[i] + d * np.cos(ang), ys[i] + d * np.sin(ang)
        else:
            cx, cy = rng.uniform(r, w - r), rng.uniform(r, h * 0.6)
        if not (r <= cx < w - r and r <= cy < h - r):
            continue
        if len(xs) and np.min(np.hypot(xs - cx, ys - cy)) < 40 + r:
            continue
        break
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(xx - cx, yy - cy)
    halo = np.where(d <= r, 1.0, np.exp(-((d - r) ** 2) / (2 * sig ** 2)))
    out = img.astype(np.float32) + (255 - img.astype(np.float32)) * halo * rng.uniform(0.6, 1.0)
    out[d <= r] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


def save(root, name, img, m, names):
    os.makedirs(f"{root}/images", exist_ok=True); os.makedirs(f"{root}/masks", exist_ok=True)
    cv2.imwrite(f"{root}/images/{name}.png", img); cv2.imwrite(f"{root}/masks/{name}.png", m); names.append(name)


def main():
    ap = argparse.ArgumentParser()
    for k in ["irstd", "adgf", "sirst", "out"]:
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sets = {k: [] for k in ["val_irstd", "val_sun", "val_bg", "sky_targets", "no_targets", "sun_synth", "sun_real"]}
    bg_names = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{a.sirst}/mixed/*.png"))
                if cv2.imread(f"{a.sirst}/annotations/masks/{os.path.basename(p)[:-4]}_pixels0.png", 0).max() == 0]
    seq = [n.rsplit("-", 1)[0] for n in bg_names]
    val_seq, nv, nt = set(), 0, 0                      # ділимо цілими серіями, порівну за кількістю кадрів
    for s in sorted(set(seq), key=lambda s: (-seq.count(s), s)):
        if nv <= nt: val_seq.add(s); nv += seq.count(s)
        else: nt += seq.count(s)
    for n in open(f"{a.adgf}/datasets/IRSTD-1K/img_idx/test_IRSTD-1K.txt").read().split():
        img = cv2.imread(f"{a.irstd}/IRSTD1k_Img/{n}.png", 0); m = cv2.imread(f"{a.irstd}/IRSTD1k_Label/{n}.png", 0)
        save(f"{a.out}/val_irstd", n, img, m, sets["val_irstd"])
        save(f"{a.out}/val_sun", n, add_sun(img, m, rng), m, sets["val_sun"])
    for p in sorted(glob.glob(f"{a.sirst}/mixed/*.png")):
        n = os.path.basename(p)[:-4]
        img = cv2.imread(p, 0); m = cv2.imread(f"{a.sirst}/annotations/masks/{n}_pixels0.png", 0)
        m = np.zeros_like(img) if m is None else m
        img, m = shrink(img, m)
        k = "sky_targets" if m.any() else ("val_bg" if n.rsplit("-", 1)[0] in val_seq else "no_targets")
        save(f"{a.out}/{k}", n, img, m, sets[k])
        if k == "val_bg":
            continue                                   # кадри валідації не йдуть у тести із сонцем
        save(f"{a.out}/sun_synth", n, add_sun(img, m, rng), m, sets["sun_synth"])
        nn, lab, st, _ = cv2.connectedComponentsWithStats((img >= 250).astype(np.uint8), 8)
        if nn > 1 and st[1:, 4].max() >= 200:
            save(f"{a.out}/sun_real", n, img, m, sets["sun_real"])
    for k, v in sets.items():
        open(f"{a.out}/{k}/list.txt", "w").write("\n".join(v)); print(k, len(v))


if __name__ == "__main__":
    main()
