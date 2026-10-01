"""Hard-negative mining for fine-tuning.

Runs a trained model over the TRAIN frames (never val / final_test), finds frames where it produces
false positives or misses 5..50 px targets, and builds a new BasicIRSTD dataset <name>-HN whose train
list repeats those frames `--repeat` extra times. Images and masks are shared via symlinks, the test
list (validation) is copied unchanged, so the fine-tuned model is still selected on the same val set.

  python mine_hard.py --model DNANet --ckpt <best.pth.tar> --mean M --std S --thr 0.5 --topk 5
Then fine-tune:
  python train.py --model_names DNANet --dataset_names THERMAL-MIX-HN --pretrained <best.pth.tar> \
      --lr 1e-4 --nEpochs 20 --save <log_ft> ...
"""
import os, json, argparse
import numpy as np, cv2, torch
import eval_thermal as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DNANet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_dir", default="./datasets")
    ap.add_argument("--name", default="THERMAL-MIX")
    ap.add_argument("--mean", type=float, required=True)
    ap.add_argument("--std", type=float, required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=2, help="extra copies of each hard frame in the new train list")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = E.build(a.model, a.ckpt, device)
    root = f"{a.dataset_dir}/{a.name}"
    train = open(f"{root}/img_idx/train_{a.name}.txt").read().split()
    hard, stats = [], dict(frames=len(train), fp_frames=0, miss_frames=0)
    for iid in train:
        img = cv2.imread(f"{root}/images/{iid}.png", 0)
        gts = E.gt_targets(cv2.imread(f"{root}/masks/{iid}.png", 0))
        prob = E.predict(net, img, a.mean, a.std, device, device == "cuda")
        dets = E.detections(prob, img, a.thr, True, 2, 80, a.topk)
        tp, fp, _ = E.match(dets, gts, 3)
        n_in = sum(g["in_scope"] for g in gts)
        if fp > 0:
            stats["fp_frames"] += 1
        if tp < n_in:
            stats["miss_frames"] += 1
        if fp > 0 or tp < n_in:
            hard.append(iid)
    new = f"{a.name}-HN"
    nroot = f"{a.dataset_dir}/{new}"
    os.makedirs(f"{nroot}/img_idx", exist_ok=True)
    for d in ["images", "masks"]:
        if not os.path.lexists(f"{nroot}/{d}"):
            os.symlink(os.path.abspath(f"{root}/{d}"), f"{nroot}/{d}")
    open(f"{nroot}/img_idx/train_{new}.txt", "w").write("\n".join(train + hard * a.repeat) + "\n")
    open(f"{nroot}/img_idx/test_{new}.txt", "w").write(open(f"{root}/img_idx/test_{a.name}.txt").read())
    stats.update(hard_frames=len(hard), new_train_size=len(train) + len(hard) * a.repeat, dataset=new)
    json.dump(stats, open(f"{nroot}/hard_mining.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
