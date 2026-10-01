"""Builds thermal45_dnanet_colab.ipynb from the scripts in this folder, so the notebook always
carries the current versions of prepare_dnanet.py, eval_thermal.py, mine_hard.py and the patch."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda f: open(f"{HERE}/{f}").read()
DOC = ""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


cells = [
md(f"""
# Кейс 4.5: детекція дрібних теплових цілей (DNA-Net)

Цей ноутбук містить весь процес: дані, навчання, оцінку за ТЗ і довчання. Кожен розділ має опис того, що робиться і чому.

**Як запустити.** Runtime → Change runtime type → **T4 GPU**. Потім Runtime → **Run all**. Дозволити доступ до Google Drive.

**Що потрібно від ТЗ і як ми це перевіряємо**

| Вимога ТЗ | Як перевіряється в ноутбуці |
|---|---|
| Кадри 640×512 | Модель працює на повному кадрі без ресайзу |
| Цілі від 5×5 px | Recall рахується окремо для цілей 5-50 px |
| ≥ 30 FPS | Замір усього пайплайна на 640×512 (розділ 4.2 і 5) |
| ≤ 1 FP на кадр | FP на кадр окремо для кожного типу сцен (розділ 5) |
| Сонце, нагріті поверхні | Окремі тестові набори: авто (FLIR), фон без цілей, синтетичне сонце |

**Зміст**
1. Налаштування
2. BasicIRSTD і патчі
3. Дані: чому ці датасети, як їх обробляємо, як качаємо
4. Навчання
5. Оцінка за ТЗ
6. Довчання: hard negatives і нові дані
7. Картинки для демо

Аналіз датасетів з графіками: {DOC}
"""),
md("""
## 1. Налаштування

**Яку модель вчити.** Основна модель **DNA-Net**. Вона дає найкраще співвідношення знайдених цілей і хибних спрацювань серед IRSTD-моделей.

Ризик у швидкості. На кадрі 640×512 DNA-Net потребує 71 GMACs (ми заміряли; у статтях пишуть 14.2G, але це для 256×256). Тому в розділі 4.2 ми міряємо FPS одразу після першого чекпойнта. Якщо менше 30 FPS, запускаємо **ALCNet** (1.9 GMACs, у 37 разів легша, вчиться 20-30 хв) як запасний варіант. Якщо DNA-Net встигає, ALCNet не потрібна.

| Модель | Обчислення на 640×512 | Роль |
|---|---|---|
| DNANet | 71.3 GMACs | основна |
| ALCNet | 1.9 GMACs | запасна, якщо не вистачить FPS |
| UIUNet | 272 GMACs | не беремо, 30 FPS не буде |

`DRIVE_DIR` це спільна папка на Google Drive. Туди пишуться дані, чекпойнти й результати. Щоб команда працювала з тими самими файлами, власник папки ділиться нею, а інші роблять Add shortcut → My Drive.
"""),
code("""
# ---- змінюйте тут ----
MODEL     = 'DNANet'     # DNANet (основна) | ALCNet (запасна, швидка)
EPOCHS    = 100          # DNANet на T4 ~1-1.5 год; ALCNet ~20-30 хв
BATCH     = 16
PATCH     = 256
DRIVE_DIR = '/content/drive/MyDrive/thermal45'
DATASET   = 'THERMAL-MIX'   # після додавання нових даних: 'THERMAL-MIX2' (розділ 6.2)
# ----------------------
!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
import os, json, glob, re
os.makedirs(DRIVE_DIR, exist_ok=True)
"""),
md("""
## 2. BasicIRSTD і патчі

**Чому BasicIRSTD, а не оригінальний репозиторій DNA-Net.** В оригінальному коді тест розтягує кадр у квадрат 256×256. Для 640×512 ціль 5 px після цього стає ~2 px, і метрики втрачають сенс. BasicIRSTD тестує на повному кадрі, а при навчанні вирізає шматки без зміни масштабу. Крім того, там в одному форматі даних є DNA-Net, ALCNet, ACM, ISTDU-Net, UIU-Net і RDIAN, тож модель міняється одним параметром.

**Що змінює патч** (`patch_basicirstd.py`):
- прибирає зламаний імпорт зі skimage, через який код падає на нових версіях;
- зберігає чекпойнт кожні 10 епох, а не 50: обрив сесії Colab забирає максимум 10 епох;
- знижує learning rate на 60% і 85% епох (в оригіналі на 200 і 300 епосі, тобто при 100 епохах ніколи);
- додає параметр `--lr` для довчання з меншим кроком;
- виправляє `--pretrained`, щоб можна було довчати з нашого чекпойнта (в оригіналі там помилка).

Також тут записуються наші скрипти: `eval_thermal.py` (метрики ТЗ) і `mine_hard.py` (hard negatives).
"""),
code("""
%cd /content
!rm -rf BasicIRSTD && git clone -q https://github.com/XinyiYing/BasicIRSTD.git
%cd /content/BasicIRSTD
"""),
code("%%writefile patch_basicirstd.py\n" + rd("patch_basicirstd.py")),
code("!python patch_basicirstd.py"),
code("%%writefile eval_thermal.py\n" + rd("eval_thermal.py")),
code("%%writefile mine_hard.py\n" + rd("mine_hard.py")),
md(f"""
## 3. Дані

### 3.1 Чому саме ці датасети

Ціль у ТЗ задана як «малорозмірна контрастна» без класу. Аналіз показав, що є два різні типи таких цілей: точкові теплові плями і дрібні об'єкти (люди, техніка). Жоден датасет не покриває обидва типи і всі завади, тому беремо кілька, кожен під свою вимогу.

| Джерело | Що закриває з ТЗ | Факти з нашого аналізу |
|---|---|---|
| HIT-UAV | дрібні люди й техніка з дрона, день і ніч | рідні 640×512; 85% цілей у 5-50 px; люди медіана 15 px |
| FLIR з авто | нагріті поверхні, наземна камера | у типовому кадрі 55% пікселів яскравіші за ціль (асфальт при 31-33 °C) |
| SIRST-v2 | точкові теплові плями; перевірка FP | контраст з фоном медіана 6.1; 510 кадрів без жодної цілі |
| синтетичне сонце | сонце в кадрі | у HIT-UAV жодного кадру з сонцем, у FLIR 2% |

Важливі факти для моделі: 39% цілей HIT-UAV і 27% цілей FLIR **холодніші за фон**, тому шукати лише яскраві плями не можна. SIRST-v1 не беремо: усі 427 його кадрів уже є в SIRST-v2.

Anti-UAV410 (дрони на фоні неба) і IRSTD-1k (точкові цілі на складному фоні) додаються другою хвилею (розділ 6.2), бо вони лежать на Google Drive/Baidu і їх треба скачати вручну.

Графіки і приклади кадрів: {DOC}

### 3.2 Як обробляємо дані (`prepare_dnanet.py`)

1. **Читання розмітки.** HIT-UAV: рамки з JSON, клас DontCare пропускаємо. FLIR: рамки з XML. SIRST-v2: готові маски.
2. **Формат.** Усі кадри переводяться у 8-біт сірий. Більші за 640×512 зменшуються зі збереженням пропорцій. Менші лишаються як є: розтягування штучно збільшило б цілі.
3. **Рамки → маски.** DNA-Net вчиться на масках, а в HIT-UAV і FLIR є лише рамки. Для кожної рамки:
    - рахуємо фон як медіану в кільці навколо рамки;
    - всередині рамки беремо пікселі, що найсильніше відрізняються від фону, поріг Otsu. Модуль різниці, тому працює і для теплих, і для холодних цілей;
    - обрізаємо еліпсом по рамці і лишаємо найбільшу пляму;
    - якщо ціль не виділяється, маска дорівнює еліпсу. Так вийшло у 27% рамок, здебільшого це великі розмиті машини понад 50 px.
4. **Дрібні цілі SIRST при зменшенні кадру** не зникають: маска зменшується з усередненням і низьким порогом.
5. **Спліт 70/15/15 по відеогрупах**, а не по кадрах. Сусідні кадри одного відео майже однакові. В офіційному спліті HIT-UAV кожна з 105 груп є і в train, і в test, тому метрики там завищені. Ми ділимо цілими групами. Кадри SIRST з цілями і без цілей діляться окремо. Кожне джерело ділиться незалежно, тому нові датасети не перемішують старі спліти.
    - **train** для навчання;
    - **val** (у BasicIRSTD це `test_THERMAL-MIX.txt`) для вибору порогу;
    - **final_test** тільки для фінальних цифр, один раз.
6. **Синтетичне сонце.** Пересвічений диск радіусом 10-35 px з ореолом у верхній половині кадру, далеко від цілей. Додаємо +10% копій train-кадрів і 150 тестових кадрів.
7. **Тестові набори окремо:** дрон (hit), авто (flir), точкові (sirst), фон без цілей (sirst_background), сонце (sun). Кожен рахується окремо, як вимагає сценарій демо.
8. **Нормалізація.** Середнє і std яскравості по train, однакові для навчання і тесту.

### 3.3 Як качається

Нічого вручну качати не треба. Клітинка нижче:
- якщо на Drive вже є `thermal_mix_dataset.tar`, просто розпаковує його (2-3 хв);
- інакше клонує HIT-UAV, SIRST-v2 і FLIR з GitHub, запускає обробку і кладе архів на Drive (~10-15 хв, лише перший раз).

Очікуваний результат: train ~4 000 кадрів, val ~1 000, final_test ~1 100.
"""),
code("%%writefile prepare_dnanet.py\n" + rd("prepare_dnanet.py")),
code("""
CACHE = f'{DRIVE_DIR}/thermal_mix_dataset.tar'
if os.path.exists(CACHE):
    !mkdir -p datasets && tar -xf {CACHE} -C datasets
else:
    %cd /content
    !git clone -q --depth 1 https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset.git hituav
    !git clone -q --depth 1 https://github.com/YimianDai/open-sirst-v2.git sirstv2
    !git clone -q --depth 1 --filter=blob:none --sparse https://github.com/sensationTI/FLIR_IR_Expansion.git flirx
    !cd flirx && git sparse-checkout set driving_data driving_annotation
    %cd /content/BasicIRSTD
    !python prepare_dnanet.py --hit /content/hituav --flir /content/flirx --sirst /content/sirstv2 --out ./datasets --name THERMAL-MIX
    !cd datasets && tar -cf {CACHE} THERMAL-MIX THERMAL-MIX-FT-* && echo saved to Drive
!ls datasets
print(open('datasets/THERMAL-MIX/summary.json').read())
"""),
md("**Перевірка масок.** Зліва кадр, зеленим маска. Якщо маски зсунуті або порожні, зупиніться і напишіть у чат команди."),
code("""
import cv2, random, numpy as np, matplotlib.pyplot as plt
R = 'datasets/THERMAL-MIX'; meta = json.load(open(f'{R}/meta.json')); random.seed(0)
ids = random.sample([k for k, m in meta.items() if m['targets']], 6)
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, iid in zip(ax.ravel(), ids):
    im = cv2.imread(f'{R}/images/{iid}.png', 0); m = cv2.imread(f'{R}/masks/{iid}.png', 0)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB); rgb[m > 0] = (0.5 * rgb[m > 0] + [0, 120, 0]).astype(np.uint8)
    a.imshow(rgb); a.set_title(iid, fontsize=8); a.axis('off')
plt.show()
"""),
md("**Нормалізація.** Рахуємо один раз по train і зберігаємо на Drive, щоб навчання, тест і демо використовували ті самі числа."),
code("""
NORM = f'{DRIVE_DIR}/norm.json'
if os.path.exists(NORM):
    MEAN, STD = json.load(open(NORM)).values()
else:
    ids = open('datasets/THERMAL-MIX/img_idx/train_THERMAL-MIX.txt').read().split()
    ms = [(im.mean(), im.std()) for im in (cv2.imread(f'datasets/THERMAL-MIX/images/{i}.png', 0).astype(np.float32) for i in ids)]
    MEAN, STD = float(np.mean([m for m, s in ms])), float(np.mean([s for m, s in ms]))
    json.dump(dict(mean=MEAN, std=STD), open(NORM, 'w'))
print('mean', MEAN, 'std', STD)
"""),
md("""
## 4. Навчання

### 4.1 Як вчиться модель

- **Вхід:** випадкові вирізки 256×256 з кадру. У половині випадків вирізка береться навколо цілі, щоб модель частіше бачила цілі.
- **Аугментації:** віддзеркалення і повороти на 90°. Масштаб не змінюємо: ціль 5 px від зменшення просто зникне.
- **Loss:** SoftIoU, стандартний для IRSTD. Він стійкий до того, що цілей у кадрі дуже мало порівняно з фоном.
- **Оптимізатор:** Adam, lr 5e-4, зниження в 10 разів на 60% і 85% епох.
- **Валідація:** кожні 10 епох друкуються mIoU і Pd/Fa на val. Pd це частка знайдених цілей, Fa це частка хибних пікселів.
- **Чекпойнти:** кожні 10 епох у `DRIVE_DIR/log/THERMAL-MIX/`. Якщо сесія обірвалась, просто Run all: навчання продовжиться з останнього чекпойнта.

З нуля, а не з готових ваг. Публічні ваги DNA-Net вчені на синтетичних точкових цілях 256×256. Наші цілі інші (люди, техніка, холодні цілі), і 100 епох з нуля цілком вистачає.
"""),
code("""
LOG = f'{DRIVE_DIR}/log'
def ckpts_of(log, ds, model):
    c = glob.glob(f'{log}/{ds}/{model}_*.pth.tar')
    return sorted(c, key=lambda p: int(re.findall(r'_(\\d+)\\.pth', p)[0]))
c = ckpts_of(LOG, DATASET, MODEL)
resume = f'--resume {c[-1]}' if c else ''
print('resume from:', resume or 'scratch')
!python train.py --model_names {MODEL} --dataset_names {DATASET} --batchSize {BATCH} --patchSize {PATCH} \\
    --nEpochs {EPOCHS} --save {LOG} --threads 2 --save_every 10 \\
    --img_norm_cfg_mean {MEAN} --img_norm_cfg_std {STD} {resume}
"""),
md("""
### 4.2 Перевірка швидкості

Запускайте, щойно з'явився перший чекпойнт (10 епох). Можна в окремій вкладці Colab, поки йде навчання. Точність тут неважлива, міряємо лише швидкість.

Міряється весь пайплайн на кадрі 640×512: нормалізація → модель (FP16) → поріг → фільтри. **Якщо FPS < 30**, поставте `MODEL = 'ALCNet'` у розділі 1 і запустіть навчання ще раз, у другій сесії або після DNA-Net.

Важливо: FPS на T4 у Colab і FPS на ноутбуці для демо різні. Фінальний замір треба робити на тому залізі, де буде демо.
"""),
code("""
c = ckpts_of(LOG, DATASET, MODEL)
if c:
    !python eval_thermal.py --model {MODEL} --ckpt {c[-1]} --subsets THERMAL-MIX-FT-sirst_background --mean {MEAN} --std {STD} --sun --half --fps --topk 5
else:
    print('ще немає чекпойнта')
"""),
md("""
## 5. Оцінка за ТЗ

Стандартні метрики IRSTD (mIoU, Pd, Fa) рахуються по пікселях і не відповідають ТЗ. Ми рахуємо так, як вимагає ТЗ:

- **Ціль** це компонента маски. **Детекція** це компонента передбачення з імовірністю вище порогу.
- **Влучання:** центр детекції всередині рамки цілі з допуском 3 px. На цілі 5×5 розмітка і прогноз легко зсуваються на 1-2 px.
- **Recall 5-50 px:** частка знайдених цілей розміром від 5 до 50 px. **FP на кадр:** детекції, що не влучили в жодну ціль.
- Детекції на цілях < 5 px або > 50 px не рахуються ні в плюс, ні в мінус.
- **Фільтри після моделі:** маска сонця (пересвічені області від 200 px + 15 px навколо), розмір детекції, top-K найвпевненіших на кадр.

**Порядок:**
1. Поріг і top-K підбираємо **тільки на val**, щоб FP на кадр було ≤ 0.7. Запас до 1, бо на нових даних буде гірше.
2. З цими налаштуваннями **один раз** рахуємо final_test окремо для кожного набору.
"""),
code("""
c = ckpts_of(LOG, DATASET, MODEL); CKPT = c[-1]; print(CKPT)
!python eval_thermal.py --model {MODEL} --ckpt {CKPT} --subsets {DATASET} --mean {MEAN} --std {STD} --sun --sweep --json {DRIVE_DIR}/sweep_{MODEL}.json
best = json.load(open(f'{DRIVE_DIR}/sweep_{MODEL}.json'))['best_under_0.7fp']; print(best)
THR, TOPK = (best['thr'], best['topk']) if best else (0.5, 3)
"""),
code("""
SUBSETS = ' '.join(f'{DATASET}-FT-{s}' for s in ['hit', 'flir', 'sirst', 'sirst_background', 'sun', 'antiuav', 'irstd']
                   if os.path.exists(f'datasets/{DATASET}-FT-{s}'))
!python eval_thermal.py --model {MODEL} --ckpt {CKPT} --subsets {SUBSETS} --mean {MEAN} --std {STD} \\
    --thr {THR} --topk {TOPK} --sun --half --fps --json {DRIVE_DIR}/final_{MODEL}.json
"""),
md("Таблиця результатів. Для кожного набору FP на кадр має бути ≤ 1, FPS ≥ 30. Recall показуємо як є, мінімуму в ТЗ немає."),
code("""
import pandas as pd
r = json.load(open(f'{DRIVE_DIR}/final_{MODEL}.json'))
df = pd.DataFrame(r['subsets']).T; df.index = [i.replace(f'{DATASET}-FT-', '') for i in df.index]
print(r['settings']); print(r['fps']); df
"""),
md("""
## 6. Довчання

Довчання продовжує навчання вже готової моделі з меншим learning rate (1e-4 замість 5e-4), а не вчить її з нуля. Є два випадки.

### 6.1 Hard negatives: якщо FP на кадр > 1 або модель пропускає цілі

1. Модель проганяється по **train**-кадрах (не по val і не по тесту, інакше метрики стануть нечесними).
2. Кадри, де вона дала хибне спрацювання або пропустила ціль, додаються в train ще двічі.
3. Модель довчається 20 епох з lr 1e-4 на цьому наборі (`THERMAL-MIX-HN`).
4. Після цього знову запускається розділ 5 з новим чекпойнтом.

Чекпойнти довчання пишуться окремо, у `DRIVE_DIR/log_ft/`, тому основна модель не перезаписується.
"""),
code("""
BEST = ckpts_of(LOG, DATASET, MODEL)[-1]
!python mine_hard.py --model {MODEL} --ckpt {BEST} --name {DATASET} --mean {MEAN} --std {STD} --thr {THR} --topk {TOPK} --repeat 2
!python train.py --model_names {MODEL} --dataset_names {DATASET}-HN --pretrained {BEST} --lr 1e-4 \\
    --batchSize {BATCH} --patchSize {PATCH} --nEpochs 20 --save {DRIVE_DIR}/log_ft --threads 2 --save_every 10 \\
    --img_norm_cfg_mean {MEAN} --img_norm_cfg_std {STD}
CKPT = ckpts_of(f'{DRIVE_DIR}/log_ft', f'{DATASET}-HN', MODEL)[-1]; print('новий чекпойнт:', CKPT)
# далі: перезапустіть дві клітинки розділу 5, але замість першого рядка з c = ... використайте цей CKPT
"""),
md("""
### 6.2 Нові дані: Anti-UAV410 і IRSTD-1k

Вони закривають те, чого бракує: дрони на фоні неба й хмар (Anti-UAV410) і точкові цілі на складному фоні (IRSTD-1k).

**Як додати:**
1. Скачати:
    - Anti-UAV410: посилання на Google Drive у README https://github.com/HwangBo94/Anti-UAV410. Розпакувати так, щоб вийшло `MyDrive/thermal45/raw/Anti-UAV410/train/<послідовність>/IR_label.json`.
    - IRSTD-1k: посилання в README https://github.com/RuiZhang97/ISNet. Розпакувати в `MyDrive/thermal45/raw/IRSTD-1k/` (всередині `IRSTD1k_Img/` і `IRSTD1k_Label/`).
2. Запустити клітинку нижче. Вона збирає новий набір `THERMAL-MIX2` зі всіма джерелами.
    - Старі джерела розподіляються між train/val/test **точно так само**, як і раніше: спліт кожного джерела незалежний, тому старий тест не протече в навчання.
    - З Anti-UAV410 береться кожен 10-й кадр відео.
3. Довчити модель з найкращого чекпойнта: 30 епох, lr 1e-4.
4. Поставити `DATASET = 'THERMAL-MIX2'` у розділі 1 і знову пройти розділ 5. У ньому використовуються набори `THERMAL-MIX2-FT-*`, там же з'являться нові `antiuav` та `irstd`.
"""),
code("""
RAW = f'{DRIVE_DIR}/raw'
!python prepare_dnanet.py --hit /content/hituav --flir /content/flirx --sirst /content/sirstv2 \\
    --antiuav {RAW}/Anti-UAV410 --irstd {RAW}/IRSTD-1k --out ./datasets --name THERMAL-MIX2
!cd datasets && tar -cf {DRIVE_DIR}/thermal_mix2_dataset.tar THERMAL-MIX2 THERMAL-MIX2-FT-*
BEST = ckpts_of(LOG, 'THERMAL-MIX', MODEL)[-1]
!python train.py --model_names {MODEL} --dataset_names THERMAL-MIX2 --pretrained {BEST} --lr 1e-4 \\
    --batchSize {BATCH} --patchSize {PATCH} --nEpochs 30 --save {LOG} --threads 2 --save_every 10 \\
    --img_norm_cfg_mean {MEAN} --img_norm_cfg_std {STD}
# якщо /content/hituav немає (дані брались з кешу), спершу виконайте клони з розділу 3.3
"""),
md("""
## 7. Картинки для демо

Зелене коло це знайдена ціль. Червоне це хибне спрацювання. Жовта рамка це пропущена ціль 5-50 px. Показуємо по 4 випадкові кадри з кожного тестового набору: дрон, сонце, точкові цілі, фон без цілей.
"""),
code("""
import torch, eval_thermal as E
net = E.build(MODEL, CKPT, 'cuda')
def show(subset, n=4):
    ids = open(f'datasets/{subset}/img_idx/test_{subset}.txt').read().split(); random.shuffle(ids)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 4.5))
    for axi, iid in zip(ax, ids[:n]):
        img = cv2.imread(f'datasets/{subset}/images/{iid}.png', 0); gts = E.gt_targets(cv2.imread(f'datasets/{subset}/masks/{iid}.png', 0))
        dets = E.detections(E.predict(net, img, MEAN, STD, 'cuda', True), img, THR, True, 2, 80, TOPK)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB); hit = set()
        for d in dets:
            h = [j for j, g in enumerate(gts) if g['x'] - 3 <= d['cx'] <= g['x'] + g['w'] + 3 and g['y'] - 3 <= d['cy'] <= g['y'] + g['h'] + 3]
            hit.update(h)
            cv2.circle(rgb, (int(d['cx']), int(d['cy'])), 10, (0, 255, 0) if h else (255, 0, 0), 2)
        for j, g in enumerate(gts):
            if g['in_scope'] and j not in hit:
                cv2.rectangle(rgb, (g['x'] - 2, g['y'] - 2), (g['x'] + g['w'] + 2, g['y'] + g['h'] + 2), (255, 220, 0), 1)
        axi.imshow(rgb); axi.set_title(iid, fontsize=8); axi.axis('off')
    plt.suptitle(subset); plt.show()
for s in ['hit', 'sun', 'sirst', 'sirst_background']:
    show(f'{DATASET}-FT-{s}')
"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "cells": cells}
json.dump(nb, open(f"{HERE}/thermal45_dnanet_colab.ipynb", "w"), ensure_ascii=False, indent=1)
print(len(cells), "cells written")
