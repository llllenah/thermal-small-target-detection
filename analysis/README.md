# Аналіз датасетів для кейсу 4.5

Скрипти, якими пораховані цифри в документі.

## Що потрібно

Python 3, `opencv-python`, `numpy`.

Поруч зі скриптами мають лежати датасети:

- `hituav/` : `git clone https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset hituav`
- `sirstv2/` : `git clone https://github.com/YimianDai/open-sirst-v2 sirstv2`
- `probe/flirx/driving_data`, `probe/flirx/driving_annotation` : з https://github.com/sensationTI/FLIR_IR_Expansion

## Запуск

1. `python eda.py` рахує для кожної цілі розмір, контраст з фоном і частку кадру, яскравішу за ціль, а для кожного кадру пересвічені області. Результат: `eda_out/eda.json`.
2. `python classic.py` проганяє класичний ланцюжок (top-hat, маска сонця, розмір, форма, контраст, top-5) і друкує Recall та FP на кадр після кожного кроку, з CLAHE і без. Результат: `eda_out/classic.json`.

## Як додати новий датасет

У `eda_loaders.py` допишіть функцію, яка повертає список кадрів у форматі:

```python
dict(ds="Назва", split="all", path="шлях/до/кадру.png",
     boxes=[(x, y, w, h, "клас"), ...], meta={})
```

Потім додайте її виклик у `all_items` в `eda.py` і в список датасетів у `classic.py`. Кадри більші за 640×512 зменшуються автоматично, менші лишаються як є.
