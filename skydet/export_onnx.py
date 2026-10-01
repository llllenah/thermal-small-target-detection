"""ADGFNet-Lite (ваги IRSTD-1k) -> adgf_lite.onnx з sigmoid на виході, вхід 1x1x512x640.
  python export_onnx.py --repo ADGFNet/ADGFNet [--fp16]"""
import sys, argparse, torch

ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--out", default="adgf_lite.onnx")
ap.add_argument("--fp16", action="store_true", help="ще й adgf_lite_fp16.onnx: ваги й обчислення FP16, вхід/вихід FP32")
a = ap.parse_args()
sys.path.insert(0, a.repo)
from model import ADGFNetLite
net = ADGFNetLite()
sd = torch.load(f"{a.repo}/checkpoints/IRSTD-1K/ADGFNetLite.pth.tar", map_location="cpu", weights_only=False)["state_dict"]
net.load_state_dict({k[6:] if k.startswith("model.") else k: v for k, v in sd.items()})


class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return torch.sigmoid(s.m(x))


torch.onnx.export(W(net).eval(), torch.randn(1, 1, 512, 640), a.out, input_names=["img"], output_names=["prob"], opset_version=17, dynamo=False)
try:
    import onnx
    from onnxsim import simplify
    m, _ = simplify(onnx.load(a.out)); onnx.save(m, a.out)
except ImportError:
    pass
print("saved", a.out)
if a.fp16:
    import onnx
    from onnxconverter_common import float16
    out16 = a.out.replace(".onnx", "_fp16.onnx")
    onnx.save(float16.convert_float_to_float16(onnx.load(a.out), keep_io_types=True), out16); print("saved", out16)
