"""ADGFNet-Lite: PyTorch -> ONNX -> TensorRT FP32 / FP16 / INT8 on a 640x512 frame.
Speed of the model on GPU + accuracy on the IRSTD-1k test set after each conversion.

  python trt_bench.py --repo ADGFNet --weights ADGFNet/checkpoints/IRSTD-1K/ADGFNetLite.pth.tar \
      --images IRSTD1k_Img --masks IRSTD1k_Label --test test.txt --train train.txt
"""
import sys, time, json, argparse
import numpy as np, cv2, torch

import bench_models as B

M, SD = 87.4661865234375, 39.71953201293945   # IRSTD-1K normalisation used by the pretrained weights


def prep(img):
    x = np.zeros((512, 640), np.float32)
    x[:img.shape[0], :img.shape[1]] = (img.astype(np.float32) - M) / SD
    return x[None, None]


def export_onnx(repo, weights, path):
    net, _ = B.load_model("adgf_lite", weights, repo)

    class W(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return torch.sigmoid(s.m(x))

    torch.onnx.export(W(net).eval(), torch.randn(1, 1, 512, 640), path, input_names=["img"], output_names=["prob"],
                      opset_version=17, dynamo=False)
    import onnx
    from onnxsim import simplify
    m, _ = simplify(onnx.load(path)); onnx.save(m, path)


def quantize_qdq(src, dst, calib_imgs):
    """INT8 with explicit Q/DQ nodes (calibrated on real train frames); TensorRT reads the scales from the graph."""
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

    class R(CalibrationDataReader):
        def __init__(s): s.it = iter([{"img": prep(i)} for i in calib_imgs])
        def get_next(s): return next(s.it, None)

    quantize_static(src, dst, R(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})   # TensorRT: zero point = 0


class TRT:
    def __init__(self, onnx_path, mode):
        import tensorrt as trt
        log = trt.Logger(trt.Logger.WARNING)
        b = trt.Builder(log)
        # Новий TensorRT (10.x+) без FP16/INT8 прапорців: точність задає сама ONNX (FP16-ваги або Q/DQ), мережа strongly typed
        typed = mode != "fp32" and not hasattr(trt.BuilderFlag, "FP16")
        net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if typed else 0)
        p = trt.OnnxParser(net, log)
        assert p.parse(open(onnx_path, "rb").read()), [p.get_error(i) for i in range(p.num_errors)]
        cfg = b.create_builder_config()
        if not typed and mode in ("fp16", "int8"): cfg.set_flag(trt.BuilderFlag.FP16)
        if not typed and mode == "int8": cfg.set_flag(trt.BuilderFlag.INT8)
        eng = trt.Runtime(log).deserialize_cuda_engine(b.build_serialized_network(net, cfg))
        self.ctx = eng.create_execution_context()
        self.inp = torch.zeros(1, 1, 512, 640, device="cuda")
        self.out = torch.zeros(1, 1, 512, 640, device="cuda")
        self.ctx.set_tensor_address("img", self.inp.data_ptr())
        self.ctx.set_tensor_address("prob", self.out.data_ptr())
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        self.inp.copy_(torch.from_numpy(x))
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.out[0, 0].cpu().numpy()


def evaluate(run, a):
    ids = open(a.test).read().split()
    tp = fp = n = 0
    for i in ids:
        img = cv2.imread(f"{a.images}/{i}.png", 0)
        gts = B.targets(cv2.imread(f"{a.masks}/{i}.png", 0))
        p = run(prep(img))[:img.shape[0], :img.shape[1]]
        t, f = B.score(B.detect(p, 0.5, 3, 1), gts); tp += t; fp += f; n += sum(g["ok"] for g in gts)
    x = prep(cv2.imread(f"{a.images}/{ids[0]}.png", 0))
    for _ in range(20): run(x)
    t0 = time.perf_counter()
    for _ in range(200): run(x)
    ms = (time.perf_counter() - t0) / 200 * 1000
    return dict(recall_5_50=round(tp / n, 3), fp_per_frame=round(fp / len(ids), 3), ms=round(ms, 2), fps=round(1000 / ms, 1))


def main():
    ap = argparse.ArgumentParser()
    for k in ["repo", "weights", "images", "masks", "test", "train"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--json", default="trt_results.json")
    a = ap.parse_args()
    export_onnx(a.repo, a.weights, "adgf_lite.onnx")
    calib = [cv2.imread(f"{a.images}/{i}.png", 0) for i in open(a.train).read().split()[:300]]
    quantize_qdq("adgf_lite.onnx", "adgf_lite_int8.onnx", calib)
    import onnx
    from onnxconverter_common import float16
    onnx.save(float16.convert_float_to_float16(onnx.load("adgf_lite.onnx"), keep_io_types=True), "adgf_lite_fp16.onnx")
    res = {"gpu": torch.cuda.get_device_name(0)}
    for name, path, mode in [("TRT FP32", "adgf_lite.onnx", "fp32"), ("TRT FP16", "adgf_lite_fp16.onnx", "fp16"),
                             ("TRT INT8", "adgf_lite_int8.onnx", "int8")]:
        try:
            res[name] = evaluate(TRT(path, mode), a)
        except Exception as e:
            res[name] = {"error": repr(e)[:300]}
        print(name, res[name], flush=True)
    json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
