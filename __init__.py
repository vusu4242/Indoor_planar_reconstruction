"""
Initialization of all files into the planes package 
(data, gt, vggt_model, ransac, aggregation, metrics, pipeline, planar_head)
"""
import importlib

from . import data, gt, vggt_model, ransac, aggregation, metrics, pipeline, planar_head
#Order for reloading: a module must be reloaded after the modules it imports from
_ORDER = ["data", "gt", "vggt_model", "ransac", "aggregation", "metrics", "pipeline", "planar_head"]


def set_root(path):
    """Sets path to the dataset"""
    data.ROOT = path
    #frames from dataset saved in memory; make None because the dataset folder changed
    data._FRAMES = None         


def reload():
    """Reread all package files from disk without restarting Colab
    (Used for period of processing code, later maybe can be deleted)"""
    root = data.ROOT
    for name in _ORDER:
        importlib.reload(globals()[name])
    data.ROOT = root
    print("перезавантажено:", ", ".join(_ORDER))
