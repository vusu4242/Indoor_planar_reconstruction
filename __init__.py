"""
Площинні примітиви на Structured3D.

Етапи (окремий файл на кожен):
    data.py         1. завантаження даних
    gt.py           2. еталонна розмітка
    vggt_model.py   3. VGGT
    ransac.py       4. пошук площин
    aggregation.py  5. очищення, злиття, фільтр меж
    metrics.py      6. метрики
    pipeline.py     7. усе разом

Використання в Colab:
    import sys; sys.path.append("/content/drive/MyDrive/AI project")
    import planes
    planes.set_root("/content/drive/MyDrive/AI project/dataset")
    model = planes.vggt_model.load_model()
    summary, frames, merges = planes.pipeline.run_all(model)

Після зміни будь-якого файлу: planes.reload()
"""
import importlib

from . import data, gt, vggt_model, ransac, aggregation, metrics, pipeline, planar_head

_ORDER = ["data", "gt", "vggt_model", "ransac", "aggregation", "metrics", "pipeline", "planar_head"]


def set_root(path):
    """Шлях до папки dataset."""
    data.ROOT = path
    data._FRAMES = None          # список кадрів іншої папки треба скласти заново


def reload():
    """Перезавантажує всі файли пакета після змін (шлях до даних зберігається,
    кеш VGGT скидається)."""
    root = data.ROOT
    for name in _ORDER:
        importlib.reload(globals()[name])
    data.ROOT = root
    print("перезавантажено:", ", ".join(_ORDER))
