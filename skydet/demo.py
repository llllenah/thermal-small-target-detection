"""Демо: папка кадрів або відео -> JSON на кожен кадр + відео з рамками, FPS і лічильником.

  python demo.py --src frames_dir_or_video.mp4 --cfg config.json --out demo_out [--temporal]
"""
import os, glob, json, time, argparse
import cv2
from detector import SkyDetector, TemporalFilter, load_cfg


def frames(src):
    if os.path.isdir(src):
        for p in sorted(glob.glob(f"{src}/*.png") + glob.glob(f"{src}/*.jpg") + glob.glob(f"{src}/*.tif*")):
            yield os.path.basename(p).rsplit(".", 1)[0], cv2.imread(p, cv2.IMREAD_UNCHANGED)
    else:
        cap, i = cv2.VideoCapture(src), 0
        while True:
            ok, f = cap.read()
            if not ok: break
            yield f"{i:06d}", f; i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--cfg"); ap.add_argument("--out", default="demo_out")
    ap.add_argument("--temporal", action="store_true"); ap.add_argument("--fps_out", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    det = SkyDetector(load_cfg(a.cfg)); tf = TemporalFilter() if a.temporal else None
    vw, js, t_all, n = None, open(f"{a.out}/detections.jsonl", "w"), 0.0, 0
    for fid, f in frames(a.src):
        if f.ndim == 3: f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        r = det(f, fid)
        if tf: r["targets"] = tf(r["targets"])
        t_all += r["time_ms"]; n += 1
        js.write(json.dumps(r) + "\n")
        vis = cv2.cvtColor(cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"), cv2.COLOR_GRAY2BGR)
        for d in r["targets"]:
            x, y, w, h = d["bbox"]
            cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 0, 255), 1)
            cv2.putText(vis, f"{d['score']:.2f}", (x, max(y - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(vis, f"{1000 / max(r['time_ms'], 1e-3):.0f} FPS  targets: {len(r['targets'])}{'  SUN' if r['sun_flag'] else ''}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        h, w = vis.shape[:2]; k = min(640 / w, 512 / h)          # відео фіксованого розміру 640x512
        vis = cv2.copyMakeBorder(cv2.resize(vis, (round(w * k), round(h * k))), 0, 512 - round(h * k), 0, 640 - round(w * k), cv2.BORDER_CONSTANT)
        if vw is None:
            vw = cv2.VideoWriter(f"{a.out}/demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps_out, (vis.shape[1], vis.shape[0]))
        vw.write(vis)
    vw and vw.release(); js.close()
    print(f"{n} кадрів, середній час {t_all / max(n, 1):.1f} мс, {1000 * n / max(t_all, 1e-3):.1f} FPS, провайдер {det.provider}")


if __name__ == "__main__":
    main()
