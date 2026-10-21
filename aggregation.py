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


def clean_planes(lab, planes, P, min_frac=0.005, open_k=3, close_k=0):
    """Ділить кожну площину на зв'язні шматки, викидає дрібні, перераховує площини.

    open_k  — розмір ядра opening (прибирає шум, тонкі смужки й містки); 0 — вимкнено.
    close_k — розмір ядра closing (заклеює маленькі дірки й щілини); 0 — вимкнено."""
    h, w = lab.shape
    out = np.zeros_like(lab)
    new = []
    for p in planes:
        m = (lab == p["label"]).astype(np.uint8)
        if open_k:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
        if close_k:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
            m &= (lab == p["label"]) | (lab == 0)            # не забираємо пікселі інших площин
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


def _plane_basis(n):
    """Дві одиничні осі, що лежать у площині з нормаллю n."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def filter_boundary(lab, planes, P, valid, max_behind=0.03, tol_mm=100, tol_rel=0.03,
                    behind_max_mm=None, footprint_margin_mm=None):
    """Залишає лише площини, за якими майже немає точок кадру (межі кімнати).

    behind_max_mm — рахувати лише точки, що лежать за площиною не далі за це число мм.
    footprint_margin_mm — рахувати лише точки, що лежать за площиною в межах самої площини
        (плюс запас у мм). None — без цих обмежень (як було спочатку)."""
    X = P[valid]
    tol = np.maximum(tol_mm, tol_rel * X[:, 2])
    out = np.zeros_like(lab)
    kept = []
    for p in planes:
        s = X @ p["n"] + p["d"]                              # відстань зі знаком (мінус = за площиною)
        is_behind = s < -tol
        if behind_max_mm is not None:
            is_behind &= s > -behind_max_mm
        if footprint_margin_mm is not None and is_behind.any():
            own = P[lab == p["label"]]                       # точки самої площини
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
