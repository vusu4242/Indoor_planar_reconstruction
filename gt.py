"""
(2) Ground Truth (GT) annotation
    Plane: n * X + d = 0 in camera coordinates (y pointing downwards), 
    with the normal facing towards the camera ($d > 0$).
    The GT mask contains only pixels that truly lie on the plane according to GT depth:
    furniture, mirrors, and views through windows are excluded.
"""
import cv2
import numpy as np

from . import data

SEM_COLORS = {1: (174, 199, 232),  
              2: (152, 223, 138),  
              3: (78, 71, 183)}    
TYPE_ID = {"wall": 1, "floor": 2, "ceiling": 3}


def plane_gt(f):
    """Layout polygons -> mask (0 = non-plane, k = k-th plane) and list of planes."""
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
    """Semantics -> 0: other, 1: wall, 2: floor, 3: ceiling (needed for the head)."""
    cls = np.zeros(f["mask"].shape[:2], np.uint8)
    for k, c in SEM_COLORS.items():
        cls[np.all(f["mask"] == c, axis=-1)] = k
    return cls


def plane_params(p):
    """Plane from annotation -> (n, d) in our coordinate system, with the normal facing towards the camera (d > 0)"""
    n = p["normal"].astype(np.float64).copy()
    n[1] = -n[1]                                   
    n /= np.linalg.norm(n)
    d = float(p["offset"])
    return (-n, -d) if d < 0 else (n, d)


def gt_frame(f, tol_mm=20, tol_rel=0.01, use_semantic=False):
    """
    Final frame GT: visible plane mask, plane parameters, GT points.
    use_semantic=False — a pixel belongs to the plane if it lies on it geometrically
    (blinds, frames, paintings flush against the wall are also counted as walls)
    use_semantic=True — the semantics must additionally match "wall/floor/ceiling"
    according to the plane type (windows, blinds, doors, paintings are filtered out).
    """
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
