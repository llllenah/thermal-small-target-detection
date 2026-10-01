"""Збирає ensemble_test.ipynb: YOLO + skydet разом, порівняння з YOLO і skydet окремо на тих самих наборах."""
import json, os
D = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(D)
nb = json.load(open(f"{D}/yolo_test.ipynb")); C = [c["source"] for c in nb["cells"]]
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def wf(path, src):
    t = open(src).read(); assert "'''" not in t
    return f"open({path!r}, 'w').write(r'''{t}''')\n"

data = C[3].replace('tqdm\n', 'tqdm onnxruntime-gpu\n', 1)
data = data.replace("for k in ['sky_targets', 'no_targets', 'sun_real']:",
                    "SVAL = {k: E.mask_items(SIRST[k]) for k in ['val_irstd', 'val_sun', 'val_bg'] if k in SIRST}\n"
                    "assert len(SVAL) == 3, 'Немає SIRST val: додайте датасет із sirst_val_part*.zip'\n"
                    "for k in ['sky_targets', 'no_targets', 'sun_real']:")
data = data.replace("print('val', len(VAL),", "print('SIRST val', {k: len(v) for k, v in SVAL.items()}, 'val', len(VAL),")

cells = [
    md("# YOLO + skydet разом: тест за ТЗ кейсу 4.5\n\n"
       "YOLO добре знаходить дрони 9-50 px, skydet (ADGFNet-Lite) знаходить точкові плями: птахи, далекі дрони. "
       "Тут вони працюють разом: беремо рамки YOLO і плями skydet, яких немає в рамках YOLO, і лишаємо одну найкращу на кадр. "
       "Пороги підбираються на val (Anti-UAV410 val + SIRST val), перевірка на тих самих test-наборах, що й у `yolo_test`. "
       "У таблиці три системи поруч: лише YOLO, лише skydet, разом.\n\n"
       "**Перед запуском**\n"
       "1. Settings: GPU T4 x2, Internet On.\n"
       "2. Add Input: модель (Output ноутбука `yolo_test` або навчання, де є `best.pt`).\n"
       "3. Add Input: датасет `sirst_parts` (тести) і новий датасет із трьох `sirst_val_part*.zip` (val SIRST і ваги skydet). "
       "Можна завантажити `sirst_val_part*.zip` новою версією в той самий датасет `sirst_parts`.\n"
       "4. `DRIVE_URL`: посилання на копію AntiUAV410.zip, яку ще не качали сьогодні.\n"
       "5. Save Version → Save & Run All. ~1.5 год. Результат: `ensemble_results.zip` в Output."),
    code(C[1] + "INT8 = False\n"),
    code(C[2].replace("print('записано')", "")
         + wf("/kaggle/working/detector.py", f"{R}/skydet/detector.py")
         + wf("/kaggle/working/fuse.py", f"{R}/yolo/fuse.py") + "print('записано')"),
    code(data),
    code(C[4] + "\n\n# skydet: ваги і конфіг із датасету sirst_val\n"
         "import onnxruntime as ort\n"
         "try: ort.preload_dlls()          # CUDA/cuDNN з PyTorch, щоб ONNX Runtime працював на GPU\n"
         "except Exception: pass\n"
         "from detector import SkyDetector, load_cfg\nimport fuse as F\n"
         "onnx = glob.glob('/kaggle/input/**/adgf_lite.onnx', recursive=True) + glob.glob('/tmp/sirst/**/adgf_lite.onnx', recursive=True)\n"
         "cfgp = glob.glob('/kaggle/input/**/skydet_config.json', recursive=True) + glob.glob('/tmp/sirst/**/skydet_config.json', recursive=True)\n"
         "assert onnx and cfgp, 'Не знайшов adgf_lite.onnx / skydet_config.json: додайте датасет із sirst_val_part*.zip'\n"
         "SCFG = load_cfg(cfgp[0]); SCFG.update(model=onnx[0], backend='cuda', topk=5, lr_thr=0.1)   # більше кандидатів, поріг підбираємо самі\n"
         "SKY = SkyDetector(SCFG); print('skydet працює на', SKY.provider)\n"
         "if SKY.provider == 'CPUExecutionProvider': print('УВАГА: skydet на CPU, буде ~2 год довше')"),
    md("## Передбачення обох моделей\nAIR, MIXED і порожні кадри беруться з передбачень на всьому test, не рахуються вдруге."),
    code("import time; t0 = time.time()\n"
         "def both(items):\n"
         "    return E.predict(model, items), F.sky_predict(SKY, items)\n"
         "PV, SV = both(VAL)\n"
         "VALSETS = {'val': (VAL, PV, SV)}\n"
         "e = [i for i, it in enumerate(VAL) if not it['gts']]; VALSETS['val_empty'] = ([VAL[i] for i in e], [PV[i] for i in e], [SV[i] for i in e])\n"
         "for k, it in SVAL.items(): VALSETS[k] = (it, *both(it))\n"
         "PY, PS = {}, {}\n"
         "PY['test_all'], PS['test_all'] = both(TEST)\n"
         "idx = {it['name']: i for i, it in enumerate(TEST)}\n"
         "for k, it in SETS.items():\n"
         "    if k == 'test_all': continue\n"
         "    if k.startswith('test_') and k != 'test_sun_synth':\n"
         "        PY[k] = [PY['test_all'][idx[x['name']]] for x in it]; PS[k] = [PS['test_all'][idx[x['name']]] for x in it]\n"
         "    else: PY[k], PS[k] = both(it)\n"
         "print(f'{(time.time() - t0) / 60:.0f} хв')"),
    md("## Пороги на val і порівняння на test\nУсі три системи: top-1 (не більше 1 FP на кадрі), FP ≤ FP_BUDGET на кожному val-наборі "
       "(Anti-UAV val, його порожні кадри, SIRST val_irstd, val_sun, val_bg). Ціль: найбільший середній Recall на Anti-UAV val і SIRST val_irstd."),
    code("NONE = [np.zeros((0, 5), np.float32)]\n"
         "def no(PP): return [NONE[0]] * len(PP)\n"
         "SYS = {}\n"
         "b = F.tune({k: (it, py, no(ps)) for k, (it, py, ps) in VALSETS.items()}, FP_BUDGET, ['val', 'val_irstd']); SYS['yolo'] = (b[1], 'y')\n"
         "b = F.tune({k: (it, no(py), ps) for k, (it, py, ps) in VALSETS.items()}, FP_BUDGET, ['val', 'val_irstd']); SYS['skydet'] = (b[1], 's')\n"
         "b = F.tune(VALSETS, FP_BUDGET, ['val', 'val_irstd']); SYS['yolo+skydet'] = (b[1], 'ys')\n"
         "for k, (cfg, _) in SYS.items(): print(k, cfg)\n"
         "def run(sysname, k):\n"
         "    cfg, use = SYS[sysname]; py, ps = PY[k], PS[k]\n"
         "    return F.score(SETS[k], py if 'y' in use else no(py), ps if 's' in use else no(ps), cfg)\n"
         "RES = {s: {k: run(s, k) for k in SETS} for s in SYS}\n"
         "rows = [dict(system=s, set=k, **{c: v[c] for c in ['frames', 'targets', 'recall', 'fp_per_frame', 'fp_max']}) for s in RES for k, v in RES[s].items()]\n"
         "TABLE = pd.DataFrame(rows)\n"
         "print(TABLE.pivot(index='set', columns='system', values='recall').round(3))\n"
         "print(TABLE.pivot(index='set', columns='system', values='fp_per_frame').round(3))\n"
         "print(pd.DataFrame({s: RES[s]['sirst_sky_targets']['recall_by_size'] for s in RES}) if 'sirst_sky_targets' in SETS else '')"),
    code("OUT = '/kaggle/working/ensemble_results'; os.makedirs(OUT, exist_ok=True)\n"
         "cfg = SYS['yolo+skydet'][0]\n"
         "for k in ['test_all', 'sirst_sky_targets', 'sirst_sun_real']:\n"
         "    if k in SETS: E.gallery(SETS[k], F.fused_list(PY[k], PS[k], cfg), dict(conf=0, topk=1, margin=0), f'{OUT}/gallery_{k}', n=8)\n"
         "print('галерея готова')"),
    md("## Швидкість усього конвеєра\nYOLO (TensorRT FP16, якщо вдалося зібрати, інакше PyTorch FP16) + skydet (ONNX Runtime CUDA) + об'єднання, batch 1."),
    code("SPEED = {}\n"
         "ym = model\n"
         "if TRT:\n"
         "    try:\n"
         "        eng = YOLO(WEIGHTS).export(format='engine', imgsz=[512, 640], batch=1, device=0, half=True)\n"
         "        ym = YOLO(eng, task='detect'); print('YOLO TensorRT FP16')\n"
         "    except Exception as e: print('TensorRT не вдався, міряю PyTorch:', repr(e)[:200])\n"
         "SPEED['yolo'] = E.speed(ym, TEST, dict(conf=SYS['yolo'][0]['ty'], topk=1, margin=0))\n"
         "SPEED['yolo+skydet'] = F.speed(ym, SKY, TEST, SYS['yolo+skydet'][0])\n"
         "pd.DataFrame(SPEED).T"),
    code("rep = dict(model=NAME, gpu=__import__('torch').cuda.get_device_name(0), fp_budget=FP_BUDGET, skydet_provider=SKY.provider,\n"
         "           configs={k: v[0] for k, v in SYS.items()}, results=RES, speed=SPEED,\n"
         "           notes=['Пороги підібрано на val (Anti-UAV410 val + SIRST val_irstd/val_sun/val_bg), test для вибору не використовувався.',\n"
         "                  'top-1: на кадрі не більше 1 FP.', 'Влучання: центр рамки в рамці цілі +-3 px; Recall для цілей 5-50 px.'])\n"
         "json.dump(rep, open(f'{OUT}/ensemble_metrics.json', 'w'), indent=1, ensure_ascii=False)\n"
         "TABLE.to_csv(f'{OUT}/summary.csv', index=False)\n"
         "shutil.make_archive('/kaggle/working/ensemble_results', 'zip', OUT)\n"
         "print(json.dumps(dict(configs=rep['configs'], speed=SPEED), indent=1))\nprint(TABLE.to_string(index=False))"),
]
json.dump({"cells": cells, "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}, open(f"{D}/ensemble_test.ipynb", "w"), ensure_ascii=False, indent=1)
print("ok", os.path.getsize(f"{D}/ensemble_test.ipynb") // 1024, "КБ")
