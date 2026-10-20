"""
Етап 2. Еталонна розмітка (GT).

Площина: n·X + d = 0 у координатах камери (y вниз), нормаль дивиться в бік камери (d > 0).
GT-маска містить лише пікселі, що справді лежать на площині за GT-глибиною:
меблі, дзеркала та вид крізь вікна туди не потрапляють.
"""
import cv2
import numpy as np

from . import data

# Кольори класів у семантиці (палітра NYU40)
SEM_COLORS = {1: (174, 199, 232),   # стіна
              2: (152, 223, 138),   # підлога
              3: (78, 71, 183)}     # стеля
TYPE_ID = {"wall": 1, "floor": 2, "ceiling": 3}


def plane_gt(f):
    """Полігони з layout -> маска (0 = не площина, k = k-та площина) і список площин."""
    lay = f["layout"]
    h, w = f["rgb"].shape[:2]
    pts = [j["coordinate"] for j in lay["junctions"]]
    mask = np.zeros((h, w), np.int32)
    planes = []
    for k, p in enumerate(lay["planes"], start=1):
        for poly in p["visible_mask"]:
            if len(poly) >= 3:
                xy = np.round(np.array([pts[i] for i in poly])).astype(np.int32)
                cv2.fillPoly(mask, [xy], k)
        planes.append({"label": k, "id": p["ID"], "type": p["type"],
                       "normal": np.array(p["normal"]), "offset": p["offset"]})
    for p in planes:
        p["pixels"] = int((mask == p["label"]).sum())
    return mask, planes


def semantic_classes(f):
    """Семантика -> 0 інше, 1 стіна, 2 підлога, 3 стеля (знадобиться для голови)."""
    cls = np.zeros(f["mask"].shape[:2], np.uint8)
    for k, c in SEM_COLORS.items():
        cls[np.all(f["mask"] == c, axis=-1)] = k
    return cls


def plane_params(p):
    """Площина з розмітки -> (n, d) у наших координатах, нормаль у бік камери (d > 0)."""
    n = p["normal"].astype(np.float64).copy()
    n[1] = -n[1]                                   # у розмітці y вгору, у нас вниз
    n /= np.linalg.norm(n)
    d = float(p["offset"])
    return (-n, -d) if d < 0 else (n, d)


def gt_frame(f, tol_mm=20, tol_rel=0.01, use_semantic=False):
    """Остаточний GT кадру: маска видимих площин, параметри площин, GT-точки.

    use_semantic=False — піксель належить площині, якщо лежить на ній геометрично
                         (жалюзі, рами, картини впритул до стіни теж рахуються стіною);
    use_semantic=True  — ще й семантика має бути «стіна/підлога/стеля» відповідно
                         до типу площини (вікна, жалюзі, двері, картини відкидаються)."""
    pmask, planes = plane_gt(f)
    P = data.backproject(f["depth"], f["cam"]["K"])
    valid = f["depth"] > 0
    tol = np.maximum(tol_mm, tol_rel * f["depth"])
    cls = semantic_classes(f) if use_semantic else None
    vis = np.zeros_like(pmask)
    for p in planes:
        p["n"], p["d"] = plane_params(p)
        on_plane = np.abs(P @ p["n"] + p["d"]) < tol
        keep = (pmask == p["label"]) & valid & on_plane
        if use_semantic:
            keep &= cls == TYPE_ID.get(p["type"], -1)
        vis[keep] = p["label"]
    return {"mask": vis, "planes": planes, "points": P, "valid": valid}
