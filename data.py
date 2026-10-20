"""
Етап 1. Завантаження даних Structured3D.

Структура на диску (усі файли одного кадру в одній папці):
    ROOT/scene_00000/room485142/view0/
        rgb.png        фото
        depth.png      глибина z, мм, uint16 (0 = немає даних)
        semantic.png   семантика (кольори NYU40)
        layout.json    розмітка площин
        camera.txt     камера Structured3D

Назва кадру в коді: scene_00000_room485142_view0.
Система координат камери: x праворуч, y вниз, z вперед; одиниці — мм.
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
    """scene_00000_room485142_view0 -> ROOT/scene_00000/room485142/view0"""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"неправильна назва кадру: {name}")
    return os.path.join(ROOT, *m.groups())


def frame_path(name, kind):
    """Шлях до файлу кадру: kind = rgb | depth | semantic | layout | camera."""
    ext = {"rgb": "rgb.png", "depth": "depth.png", "semantic": "semantic.png",
           "layout": "layout.json", "camera": "camera.txt"}[kind]
    return os.path.join(frame_dir(name), ext)


def list_frames():
    """Назви всіх кадрів (папки, де є rgb.png)."""
    names = []
    for p in glob.glob(os.path.join(ROOT, "scene_*", "room*", "view*", "rgb.png")):
        view_dir = os.path.dirname(p)
        room_dir = os.path.dirname(view_dir)
        scene_dir = os.path.dirname(room_dir)
        names.append("_".join(os.path.basename(d) for d in (scene_dir, room_dir, view_dir)))
    return sorted(names, key=lambda n: (n.rsplit("_view", 1)[0], int(n.rsplit("_view", 1)[1])))


def list_rooms():
    """Назви кімнат: scene_xxxxx_roomyyyy."""
    return sorted({n.rsplit("_view", 1)[0] for n in list_frames()})


def room_frames(room):
    """Кадри однієї кімнати (view0, view1, ...)."""
    return [n for n in list_frames() if n.startswith(room + "_view")]


def parse_camera(path, h, w):
    """Камера Structured3D -> поворот R, позиція t (мм), матриця K."""
    v = np.loadtxt(path)
    pos, toward, up = v[0:3], v[3:6], v[6:9]
    xfov, yfov = v[9], v[10]                      # половини кутів огляду, радіани
    W = toward / np.linalg.norm(toward)           # вперед
    U = np.cross(W, up); U /= np.linalg.norm(U)   # праворуч
    V = np.cross(W, U)                            # вниз
    R = np.stack([U, V, W])                       # світ -> камера: X_cam = R @ (X - t)
    K = np.array([[w / 2 / np.tan(xfov), 0, w / 2],
                  [0, h / 2 / np.tan(yfov), h / 2],
                  [0, 0, 1]])
    return {"R": R, "t": pos, "K": K}


def load_frame(name):
    """Усе для одного кадру за його назвою."""
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
    """Глибина z (мм) -> 3D-точки в координатах камери (H, W, 3)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    d = depth.astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.stack([x * d, y * d, d], -1)


def random_colors(mask, seed=0):
    """Кожен номер у масці — свій колір, 0 — чорний (для візуалізації)."""
    pal = np.random.default_rng(seed).integers(40, 255, (int(mask.max()) + 1, 3)).astype(np.uint8)
    pal[0] = 0
    return pal[mask]
