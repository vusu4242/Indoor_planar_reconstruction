"""
(1) Loading Structured3D data

Structure on drive (all files of one frame are in one folder):
    ROOT/scene_00000/room485142/view0/
        rgb.png:
        depth.png     
        semantic.png 
        layout.json   
        camera.txt     

Name of frame in code: scene_00000_room485142_view0.
Camera coordinate system: x-right, y-down, z-forward;; measure - mm
"""

import glob
import json
import os
import re

import cv2
import numpy as np

ROOT = "/content/drive/MyDrive/AI project/dataset"

_NAME = re.compile(r"^(scene_\d+)_(room\d+)_(view\d+)$")


def frame_dir(name):
    """
    scene_00000_room485142_view0 -> ROOT/scene_00000/room485142/view0
    """
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"неправильна назва кадру: {name}")
    return os.path.join(ROOT, *m.groups())


def frame_path(name, kind):
    """
    One view file path: kind = rgb | depth | semantic | layout | camera
    """
    ext = {"rgb": "rgb.png", "depth": "depth.png", "semantic": "semantic.png",
           "layout": "layout.json", "camera": "camera.txt"}[kind]
    return os.path.join(frame_dir(name), ext)


INDEX_FILE = "index.txt"    
_FRAMES = None               


def _sort(names):
    """
    Sorts by room and then by view (view2 < view10)
    """
    return sorted(set(names), key=lambda n: (n.rsplit("_view", 1)[0], int(n.rsplit("_view", 1)[1])))


def _scan(pattern):
    """
    Returns names of views that include rgb
    """
    names = []
    for p in glob.glob(pattern):
        view_dir = os.path.dirname(p)
        room_dir = os.path.dirname(view_dir)
        scene_dir = os.path.dirname(room_dir)
        names.append("_".join(os.path.basename(d) for d in (scene_dir, room_dir, view_dir)))
    return names


def list_frames(scenes=None, refresh=False):
    global _FRAMES
    if scenes is not None:
        names = []
        for sc in scenes:
            names += _scan(os.path.join(ROOT, sc, "room*", "view*", "rgb.png"))
        return _sort(names)

    if _FRAMES is not None and not refresh:
        return _FRAMES
    index = os.path.join(ROOT, INDEX_FILE)
    if os.path.exists(index) and not refresh:
        with open(index) as f:
            _FRAMES = _sort(line.strip() for line in f if line.strip())
        return _FRAMES

    print("Forming list of views...")
    _FRAMES = _sort(_scan(os.path.join(ROOT, "scene_*", "room*", "view*", "rgb.png")))
    if not _FRAMES:
        print("No views found")
        _FRAMES = None
        return []
    try:
        with open(index, "w") as f:
            f.write("\n".join(_FRAMES))
        print(f"{len(_FRAMES)} views, list saved in {index}")
    except OSError:
        print(f"{len(_FRAMES)} views, index.txt not saved, no edit rights")
    return _FRAMES

def load_split(name):
    """Scenes from split.json: name = train | val | test."""
    with open(os.path.join(os.path.dirname(__file__), "split.json")) as f:
        return json.load(f)[name]

def list_scenes():
    """Names of scenes on drive"""
    return sorted(d for d in os.listdir(ROOT) if d.startswith("scene_"))


def list_rooms(scenes=None):
    """Names of rooms"""
    return sorted({n.rsplit("_view", 1)[0] for n in list_frames(scenes)})


def room_frames(room):
    """One room views"""
    scene, rm = room.split("_room")
    return _sort(_scan(os.path.join(ROOT, scene, "room" + rm, "view*", "rgb.png")))


def parse_camera(path, h, w):
    """Camers Structured3D -> rotation R, position t (mm), matrix K."""
    v = np.loadtxt(path)
    pos, toward, up = v[0:3], v[3:6], v[6:9]
    xfov, yfov = v[9], v[10]                      
    W = toward / np.linalg.norm(toward)           
    U = np.cross(W, up); U /= np.linalg.norm(U)   
    V = np.cross(W, U)                            
    R = np.stack([U, V, W])                       
    K = np.array([[w / 2 / np.tan(xfov), 0, w / 2],
                  [0, h / 2 / np.tan(yfov), h / 2],
                  [0, 0, 1]])
    return {"R": R, "t": pos, "K": K}


def load_frame(name):
    """Everything for one view by name"""
    rgb = cv2.cvtColor(cv2.imread(frame_path(name, "rgb")), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    depth = cv2.imread(frame_path(name, "depth"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(frame_path(name, "semantic"), cv2.IMREAD_UNCHANGED)
    if mask is not None and mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    with open(frame_path(name, "layout")) as f:
        layout = json.load(f)
    cam = parse_camera(frame_path(name, "camera"), h, w)
    return {"name": name, "rgb": rgb, "depth": depth, "mask": mask,
            "layout": layout, "cam": cam}


def backproject(depth, K):
    """Depth z (mm) -> 3D-dots in camera coordinates (H, W, 3)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    d = depth.astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.stack([x * d, y * d, d], -1)

def smooth_depth(z, d=7, sigma_rel=0.02, sigma_space=5):
    """
    Bilateral filter on depth. Smoothes noise inside surfaces, keeps sharp depth edges.
    d: neighbourhood size in pixels (0 = off);
    sigma_rel: depth differences below this fraction of depth (0.02 = 2 %) count as noise;
    works on log(depth), so the threshold is the same for near and far points;
    sigma_space: how fast the influence of neighbours decreases with distance (pixels)
    
    """
    if not d:
        return z
    zf = np.log(np.maximum(z, 1e-6)).astype(np.float32)
    zf = cv2.bilateralFilter(zf, d, float(sigma_rel), float(sigma_space))
    return np.exp(zf.astype(np.float64))


def random_colors(mask, seed=0):
    """Each number in the mask has its own color; 0 is black (for visualization)."""
    pal = np.random.default_rng(seed).integers(40, 255, (int(mask.max()) + 1, 3)).astype(np.uint8)
    pal[0] = 0
    return pal[mask]
