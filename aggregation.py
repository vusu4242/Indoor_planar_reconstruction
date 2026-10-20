"""
Етап 5. Агрегація площин:
    clean_planes     — розбиття на зв'язні шматки, викидання дрібних;
    merge_coplanar   — злиття шматків однієї площини всередині кадру;
    filter_boundary  — залишаємо лише межі кімнати (за ними майже немає точок);
    merge_items      — злиття однакових площин між кадрами (у світових координатах).
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .ransac import fit_plane_pca


def _union_find(n, pairs_ok):
    """Групує елементи 0..n-1: i та j в одній групі, якщо pairs_ok(i, j) (транзитивно)."""
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
    """Нова маска і площини з груп старих номерів (площини перераховуються через PCA)."""
    out = np.zeros_like(lab)
    new = []
    for labels in groups_of_labels:
        sel = np.isin(lab, labels)
        n_, d_ = fit_plane_pca(P[sel])
        new.append({"label": len(new) + 1, "n": n_, "d": d_, "pixels": int(sel.sum())})
        out[sel] = len(new)
    return out, new


def clean_planes(lab, planes, P, min_frac=0.005):
    """Ділить кожну площину на зв'язні шматки, викидає дрібні, перераховує площини."""
    h, w = lab.shape
    out = np.zeros_like(lab)
    new = []
    kernel = np.ones((3, 3), np.uint8)
    for p in planes:
        m = (lab == p["label"]).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)      # рвемо тонкі «містки»
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
    """Зливає шматки однієї площини в межах кадру (однакові нормаль і відстань)."""
    def ok(i, j):
        a, b = planes[i], planes[j]
        ang = np.degrees(np.arccos(np.clip(a["n"] @ b["n"], -1, 1)))
        return ang < ang_thr and abs(a["d"] - b["d"]) < max(d_abs, d_rel * (a["d"] + b["d"]) / 2)
    roots = _union_find(len(planes), ok)
    groups = {}
    for i, r in enumerate(roots):
        groups.setdefault(r, []).append(planes[i]["label"])
    return _relabel(lab, groups.values(), P)


def filter_boundary(lab, planes, P, valid, max_behind=0.03, tol_mm=100, tol_rel=0.03,
                    behind_max_mm=None):
    """Залишає лише площини, за якими майже немає точок кадру (межі кімнати).

    behind_max_mm — рахувати лише точки, що лежать за площиною НЕ далі за це число мм.
        Під стільницею підлога (~0.7 м), за фасадом шафи стіна (~0.5 м) — вони рахуються.
        Вулиця крізь вікно чи сусідня кімната крізь скло (метри) — ні, тож стіни
        з вікнами не відкидаються. None — рахувати всі точки за площиною (як раніше)."""
    X = P[valid]
    tol = np.maximum(tol_mm, tol_rel * X[:, 2])
    out = np.zeros_like(lab)
    kept = []
    for p in planes:
        s = X @ p["n"] + p["d"]                              # відстань зі знаком (мінус = за площиною)
        is_behind = s < -tol
        if behind_max_mm is not None:
            is_behind &= s > -behind_max_mm
        behind = np.mean(is_behind)                          # частка точок за площиною
        if behind <= max_behind:
            kept.append({**p, "label": len(kept) + 1, "behind": float(behind)})
            out[lab == p["label"]] = len(kept)
    return out, kept


def same_plane(a, b, ang_thr=5, d_thr=100, touch_mm=200):
    """Чи це та сама площина на різних кадрах (у світових координатах)."""
    na, nb, db = a["n"], b["n"], b["d"]
    if na @ nb < 0:                                          # нормалі в різні боки -> вирівнюємо
        nb, db = -nb, -db
    if np.degrees(np.arccos(np.clip(na @ nb, -1, 1))) > ang_thr:
        return False
    if abs(a["d"] - db) > d_thr:
        return False
    dist, _ = cKDTree(b["pts"]).query(a["pts"], k=1)
    return np.percentile(dist, 5) < touch_mm                 # точки двох площин торкаються


def merge_items(items, **kw):
    """Злиття площин між кадрами. Повертає номер групи (1, 2, ...) для кожної площини."""
    roots = _union_find(len(items), lambda i, j: items[i]["frame"] != items[j]["frame"]
                        and same_plane(items[i], items[j], **kw))
    ids = {}
    return [ids.setdefault(r, len(ids) + 1) for r in roots]


def to_world(planes, lab, P, R, t, max_pts=2000, seed=0):
    """Площини кадру -> світові координати (для злиття між кадрами).
    Камера: X_cam = R·X_world + t  =>  n_w = Rᵀn,  d_w = d + n·t,  X_world = Rᵀ(X_cam − t)."""
    rng = np.random.default_rng(seed)
    out = []
    for p in planes:
        pts = P[lab == p["label"]]
        pts = pts[rng.choice(len(pts), min(max_pts, len(pts)), replace=False)]
        out.append({"label": p["label"], "n": R.T @ p["n"], "d": p["d"] + p["n"] @ t,
                    "pts": (pts - t) @ R})
    return out
