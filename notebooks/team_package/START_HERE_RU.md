# Anti-UAV410 SKY: сборка для команды

Нужны Python 3.10+ (проверено на 3.12), эта папка из 6 файлов и ваш локальный архив **AntiUAV410.zip**.
Положите архив рядом с файлами team_package и откройте PowerShell в этой папке.
analysis/, review/ и исходные решения для сборки не нужны. Архив читается, не изменяется.

## Подготовка (один раз)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 1. Проверка ссылок в архиве
```powershell
.\.venv\Scripts\python.exe build_antiuav_yolo.py --source "AntiUAV410.zip" --manifest "selected_frames.csv.gz" --output "AntiUAV410_SKY_CHECK" --check-only
if ($LASTEXITCODE -ne 0) { throw "Preflight failed; inspect AntiUAV410_SKY_CHECK/preflight_errors.csv" }
```

Ожидается METADATA_PASS и 141880/141880 совпадений, 0 ошибок.
Проверяются ZIP-пути и семантика labels; сами JPEG на этом шаге не декодируются.
В preflight_report.json указана оценка объёма выбранных изображений; нужен дополнительный запас места для labels и файловой системы.

## 2. Полная сборка
```powershell
.\.venv\Scripts\python.exe build_antiuav_yolo.py --source "AntiUAV410.zip" --manifest "selected_frames.csv.gz" --output "AntiUAV410_SKY_YOLO"
if ($LASTEXITCODE -ne 0) { throw "Build failed; inspect AntiUAV410_SKY_YOLO/build_errors.csv" }
```

**CHECK и YOLO — разные новые/пустые папки.** Check-only тоже пишет отчёты; builder не разрешает повторное использование непустого output. При повторе укажите новые имена, не удаляйте результаты автоматически.
Ожидается COMPLETE: 141880/141880, 0 ошибок. Будут images/train|val|test, labels/train|val|test, data.yaml и отчёты.
Builder v1.1.0 скопирован без изменений; его встроенный help может содержать исторический пример имени архива. Рабочие команды выше используют правильное имя.

## 3. Проверка результата
```powershell
.\.venv\Scripts\python.exe verify_dataset.py --dataset "AntiUAV410_SKY_YOLO" --manifest "selected_frames.csv.gz" --fingerprint "dataset_fingerprint.json"
if ($LASTEXITCODE -ne 0) { throw "Verification failed" }
```

Ожидается PASS: train 98728; val 21601; test 21551.
Проверяются SHA256 manifest, fingerprint counts, точные пары/имена, отсутствие лишних файлов, labels и координаты, JPEG заголовки/размеры, data.yaml и build_report.
Полное декодирование JPEG выполняет builder. После переноса dataset обновите path в data.yaml.

## Границы проверки
RAW validation на машине подготовки: **PENDING_RAW_VALIDATION**.
Ничего не скачивалось; готовые RAW JPEG не передаются.
Позитивы: AIR/MIXED, обе стороны >=5 px, один класс thermal_target; GROUND/REVIEW positives отсутствуют.
Negatives определены exist=0; hard_negative — proxy по preview/metadata, не результаты модели.
14 sequences исключены как REVIEW. Preview — девять выборочных кадров, не полная проверка каждой аннотации.
AIR 5–8 в val/test очень мало (6/2); нужны дополнительные независимые данные для надёжной оценки.
MIXED — 77.5% позитивов; negatives — 4.84% всей выборки.
Fingerprint sequence_decisions_sha256 сохраняет происхождение; CSV решений не входит в минимальный пакет и здесь не пересчитывается.
Скорость >=30 FPS, <=1 FP/frame и реальная сложность test проверяются отдельным обучением/оценкой.

