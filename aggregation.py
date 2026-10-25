"""
(5) Planes aggregation

    clean_planes: splits into connected components and discards small fragments
    merge_coplanar: merges coplanar fragments within the same frame
    filter_boundary: keeps only room boundaries (filtering out planes with almost no points behind them)
    fill_holes: morphological closing on masks at the end
    merge_items: merges identical planes across multiple views (in world coordinates)

"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .ransac import fit_plane_pca


def fill_holes(lab, k=5):
    """
    Morphological closing for each plane, k - kernel size.
    Fills pixels that do not belong to any plane (background 0) and does not recompute plane parameters,
    so geometry (normals, offsets) remains unaffected—only the masks are updated.
    """
    out = lab.copy()
    kernel = np.ones((k, k), np.uint8)
    for lbl in np.unique(lab[lab > 0]):
        m = cv2.morphologyEx((lab == lbl).astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        out[m & (out == 0)] = lbl
    return out


def _union_find(n, pairs_ok):
    """Groups elements if pairs_ok(i, j)"""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if pairs_ok(i, j):
                parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def _relabel(lab, groups_of_labels, P):
    """New mask and planes from old numbers (using PCA)"""
    out = np.zeros_like(lab)
    new = []
    for labels in groups_of_labels:
        sel = np.isin(lab, labels)
        n_, d_ = fit_plane_pca(P[sel])
        new.append({"label": len(new) + 1, "n": n_, "d": d_, "pixels": int(sel.sum())})
        out[sel] = len(new)
    return out, new


def clean_planes(lab, planes, P, min_frac=0.005, open_k=3):
    """
    Splits each plane into connected components, discards small fragments, and recalculates plane parameters
    open_k — size of the opening kernel (removes noise, thin strips, and bridges), 0 — disabled
    Sizes are in VGGT mask pixels (294×518): 3 here ~ 7 pixels on a full 1280×720 frame
    (Closing is handled separately in fill_holes at the end of the pipeline.)
    """
    h, w = lab.shape
    out = np.zeros_like(lab)
    new = []
    for p in planes:
        m = (lab == p["label"]).astype(np.uint8)
        if open_k:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
        nc, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for c in range(1, nc):
            if stats[c, cv2.CC_STAT_AREA] < min_frac * h * w:
                continue
            sel = cc == c
            n_, d_ = fit_plane_pca(P[sel])
            new.append({"label": len(new) + 1, "n": n_, "d": d_, "pixels": int(sel.sum())})
            out[sel] = len(new)
    return out, new


def merge_coplanar(lab, planes, P, ang_thr=5, d_abs=60, d_rel=0.03):
    """
    Merges coplanar fragments within the same frame (matching normal and offset).
    """
    def ok(i, j):
        a, b = planes[i], planes[j]
        ang = np.degrees(np.arccos(np.clip(a["n"] @ b["n"], -1, 1)))
        return ang < ang_thr and abs(a["d"] - b["d"]) < max(d_abs, d_rel * (a["d"] + b["d"]) / 2)
    roots = _union_find(len(planes), ok)
    groups = {}
    for i, r in enumerate(roots):
        groups.setdefault(r, []).append(planes[i]["label"])
    return _relabel(lab, groups.values(), P)


def _plane_basis(n):
    """Two unit axes lying in the plane with normal $n$."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def filter_boundary(lab, planes, P, valid, max_behind=0.03, tol_mm=100, tol_rel=0.03,
                    behind_max_mm=None, footprint_margin_mm=None):
    """Retains only planes that have almost no frame points behind them (room boundaries).

    behind_max_mm — counts only points lying behind the plane no further than this value in mm.
    footprint_margin_mm — counts only points lying behind the plane WITHIN the plane's footprint
    (plus a margin in mm). None — disables these constraints."""
    X = P[valid]
    tol = np.maximum(tol_mm, tol_rel * X[:, 2])
    out = np.zeros_like(lab)
    kept = []
    for p in planes:
        s = X @ p["n"] + p["d"]                            
        is_behind = s < -tol
        if behind_max_mm is not None:
            is_behind &= s > -behind_max_mm
        if footprint_margin_mm is not None and is_behind.any():
            own = P[lab == p["label"]]                      
            u, v = _plane_basis(p["n"])
            c = own.mean(0)
            ou, ov = (own - c) @ u, (own - c) @ v
            lo_u, hi_u = np.percentile(ou, [2, 98])
            lo_v, hi_v = np.percentile(ov, [2, 98])
            m = footprint_margin_mm
            idx = np.flatnonzero(is_behind)
            B = X[idx] - c
            bu, bv = B @ u, B @ v                            
            inside = (bu > lo_u - m) & (bu < hi_u + m) & (bv > lo_v - m) & (bv < hi_v + m)
            is_behind = np.zeros_like(is_behind)
            is_behind[idx[inside]] = True                    
        behind = np.mean(is_behind)                          
        if behind <= max_behind:
            kept.append({**p, "label": len(kept) + 1, "behind": float(behind)})
            out[lab == p["label"]] = len(kept)
    return out, kept


def same_plane(a, b, ang_thr=5, d_thr=100, touch_mm=200):
    """Checks if this is the same plane across different frames (in world coordinates)."""
    na, nb, db = a["n"], b["n"], b["d"]
    if na @ nb < 0:                                        
        nb, db = -nb, -db
    if np.degrees(np.arccos(np.clip(na @ nb, -1, 1))) > ang_thr:
        return False
    if abs(a["d"] - db) > d_thr:
        return False
    dist, _ = cKDTree(b["pts"]).query(a["pts"], k=1)
    return np.percentile(dist, 5) < touch_mm                 


def merge_items(items, **kw):
    """Merges planes across frames. Returns the group ID (1, 2, ...) for each plane."""
    roots = _union_find(len(items), lambda i, j: items[i]["frame"] != items[j]["frame"]
                        and same_plane(items[i], items[j], **kw))
    ids = {}
    return [ids.setdefault(r, len(ids) + 1) for r in roots]


def to_world(planes, lab, P, R, t, max_pts=2000, seed=0):
    """Transforms frame planes to world coordinates (for cross-view merging).
    Camera convention: 
    X_cam = R·X_world + t  =>  n_w = Rᵀn,  d_w = d + n·t,  X_world = Rᵀ(X_cam − t)"""
    rng = np.random.default_rng(seed)
    out = []
    for p in planes:
        pts = P[lab == p["label"]]
        pts = pts[rng.choice(len(pts), min(max_pts, len(pts)), replace=False)]
        out.append({"label": p["label"], "n": R.T @ p["n"], "d": p["d"] + p["n"] @ t,
                    "pts": (pts - t) @ R})
    return out
