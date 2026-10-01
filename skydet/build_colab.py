"""Збирає skydet_colab.ipynb: уся система + фінальний прогін на GPU Colab."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
wf = lambda path, src: code(f"%%writefile {path}\n" + open(src).read())
cells = [
    md("# Система детекції цілей на небі: фінальний прогін на GPU\n\nRuntime → Change runtime type → **T4 GPU** → Run all. Близько 25 хв.\n\n"
       "1. Код системи (`skydet/`): детектор, фільтри, тестові набори, оцінка, демо.\n"
       "2. Експорт ADGFNet-Lite в ONNX FP32 і FP16.\n3. Тестові набори: небо з цілями, без цілей, сонце (синтетичне), реальні пересвічення.\n"
       "4. Підбір порогу і фільтра хибних плям на валідації (IRSTD-1k test, він же з сонцем, половина серій SIRST-v2 без цілей), FP ≤ 0.7 на кадр.\n"
       "5. Метрики ТЗ на кожному наборі окремо + абляція фільтрів + FPS усього конвеєра.\n6. TensorRT FP32 / FP16 / INT8.\n7. Демо-відео з рамками і FPS."),
    code("!nvidia-smi -L\n%cd /content\n"
         "!git clone -q --depth 1 https://github.com/kuaileBenbi/ADGFNet.git\n"
         "!git clone -q --depth 1 https://github.com/UsenkoAnastasiya/filtered-data-IRSTD1k.git irstd\n"
         "!git clone -q --depth 1 https://github.com/YimianDai/open-sirst-v2.git sirstv2\n"
         "!pip install -q onnx onnxsim onnxruntime-gpu onnxconverter-common tensorrt scikit-learn\n!mkdir -p /content/skydet"),
    wf("/content/skydet/detector.py", f"{D}/detector.py"),
    wf("/content/skydet/make_testsets.py", f"{D}/make_testsets.py"),
    wf("/content/skydet/evaluate.py", f"{D}/evaluate.py"),
    wf("/content/skydet/export_onnx.py", f"{D}/export_onnx.py"),
    wf("/content/skydet/demo.py", f"{D}/demo.py"),
    wf("/content/bench_models.py", f"{D}/../bench_models.py"),
    wf("/content/trt_bench.py", f"{D}/../trt_bench.py"),
    code("%cd /content/skydet\n!python export_onnx.py --repo /content/ADGFNet/ADGFNet --fp16 2>&1 | grep saved\n"
         "!python make_testsets.py --irstd /content/irstd --adgf /content/ADGFNet/ADGFNet --sirst /content/sirstv2 --out /content/testsets\n"
         "import onnxruntime as ort; print(ort.get_available_providers())"),
    md("Підбір порогу, top-K і фільтра хибних плям на валідації. Результат пишеться в `config.json`."),
    code("!python evaluate.py --data /content/testsets --tune --backend cuda | tail -1\n!cat config.json"),
    md("Метрики ТЗ на тестових наборах (модель їх не бачила), FP32 і FP16. Абляція показує внесок кожного фільтра."),
    code("import json, pandas as pd\n"
         "!python evaluate.py --data /content/testsets --cfg config.json --backend cuda --ablation --json res_fp32.json > /dev/null 2>&1\n"
         "!python evaluate.py --data /content/testsets --cfg config.json --backend cuda --model adgf_lite_fp16.onnx --json res_fp16.json > /dev/null 2>&1\n"
         "for f in ['res_fp32.json', 'res_fp16.json']:\n"
         "    r = json.load(open(f)); print(f, r['speed']); display(pd.DataFrame(r['sets']).T)\n"
         "ab = json.load(open('res_fp32.json'))['ablation']\n"
         "display(pd.DataFrame({(k, s): {'recall': v[s]['recall'], 'fp': v[s]['fp_per_frame']} for k, v in ab.items() for s in v}).T)"),
    md("TensorRT: швидкість самої моделі і Recall/FP на IRSTD-1k test після кожної конвертації."),
    code("%cd /content\n!python /content/trt_bench.py --repo /content/ADGFNet/ADGFNet --weights /content/ADGFNet/ADGFNet/checkpoints/IRSTD-1K/ADGFNetLite.pth.tar "
         "--images /content/irstd/IRSTD1k_Img --masks /content/irstd/IRSTD1k_Label --test /content/ADGFNet/ADGFNet/datasets/IRSTD-1K/img_idx/test_IRSTD-1K.txt "
         "--train /content/ADGFNet/ADGFNet/datasets/IRSTD-1K/img_idx/train_IRSTD-1K.txt 2>&1 | grep '^TRT'\n"
         "pd.DataFrame(json.load(open('/content/trt_results.json'))).T"),
    md("Демо: 300 кадрів із сонцем → відео з рамками і FPS, JSON на кожен кадр."),
    code("%cd /content/skydet\n!mkdir -p /content/demo_frames && ls /content/testsets/sun_synth/images | head -300 | xargs -I{} cp /content/testsets/sun_synth/images/{} /content/demo_frames/\n"
         "!python demo.py --src /content/demo_frames --cfg config.json --out /content/demo_out\n"
         "!ffmpeg -loglevel error -y -i /content/demo_out/demo.mp4 -vcodec libx264 /content/demo_out/demo_h264.mp4\n"
         "from IPython.display import Video; Video('/content/demo_out/demo_h264.mp4', embed=True, width=640)"),
    code("# Зведення одним рядком (для звіту)\nprint(json.dumps(dict(fp32=json.load(open('res_fp32.json'))['sets'], speed32=json.load(open('res_fp32.json'))['speed'],\n"
         "    speed16=json.load(open('res_fp16.json'))['speed'], trt=json.load(open('/content/trt_results.json')))))"),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(f"{D}/../skydet_colab.ipynb", "w"), ensure_ascii=False, indent=1)
print("ok")
