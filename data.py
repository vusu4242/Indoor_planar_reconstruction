"""
(1) Завантаження даних

Структура (однакова назва файлу кадру в усіх папках):
    ROOT/images/<кадр>.png   фото
    ROOT/depths/<кадр>.png   глибина z, мм, uint16 (0 = немає даних)
    ROOT/masks/<кадр>.png    семантика (кольори NYU40)
    ROOT/layouts/<кадр>.json розмітка площин
    ROOT/cameras/<кадр>.txt  камера Structured3D

Система координат камери: x праворуч, y вниз, z вперед; одиниці — мм.
"""
import json
import os

import cv2
import numpy as np

ROOT = "/content/drive/MyDrive/AI project/dataset"


def list_frames():
    """Назви всіх кадрів (без розширення)."""
    return sorted(os.path.splitext(f)[0] for f in os.listdir(f"{ROOT}/images"))


def list_rooms():
    """Назви кімнат: scene_xxxxx_roomyyyy"""
    return sorted({n.rsplit("_view", 1)[0] for n in list_frames()})


def room_frames(room):
    """Кадри однієї кімнати"""
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
    """Усе для одного кадру за його назвою"""
    rgb = cv2.cvtColor(cv2.imread(f"{ROOT}/images/{name}.png"), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    depth = cv2.imread(f"{ROOT}/depths/{name}.png", cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(f"{ROOT}/masks/{name}.png", cv2.IMREAD_UNCHANGED)
    if mask is not None and mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    with open(f"{ROOT}/layouts/{name}.json") as f:
        layout = json.load(f)
    cam = parse_camera(f"{ROOT}/cameras/{name}.txt", h, w)
    return {"name": name, "rgb": rgb, "depth": depth, "mask": mask,
            "layout": layout, "cam": cam}


def backproject(depth, K):
    """Глибина z (мм) -> 3D-точки в координатах камери (H, W, 3)"""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    d = depth.astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.stack([x * d, y * d, d], -1)


def random_colors(mask, seed=0):
    """Кожен номер у масці — свій колір, 0 — чорний (для візуалізації)"""
    pal = np.random.default_rng(seed).integers(40, 255, (int(mask.max()) + 1, 3)).astype(np.uint8)
    pal[0] = 0
    return pal[mask]
