"""
Етап 4. Пошук площин у хмарі точок.

Два порядки:
    sequential_ransac   — RANSAC по всьому кадру (знайшли площину -> прибрали точки -> далі);
    segment_then_ransac — спершу різати кадр на області за нормалями й глибиною VGGT,
                          потім RANSAC окремо в кожній області.
"""
import cv2
import numpy as np


def fit_plane_pca(X):
    """Найкраща площина через точки: n·X + d = 0, нормаль у бік камери (d > 0)."""
    c = X.mean(0)
    n = np.linalg.svd(X - c, full_matrices=False)[2][2]
    d = -n @ c
    return (-n, -d) if d < 0 else (n, d)


def sequential_ransac(P, z, valid, max_planes=12, iters=300, min_frac=0.01,
                      tol_mm=30, tol_rel=0.02, seed=0):
    """Шукає площини одна за одною: знайшли -> прибрали її точки -> шукаємо далі.

    P: (H, W, 3) точки в мм, z: (H, W) глибина в мм, valid: (H, W) які пікселі брати.
    Повертає маску (0 = не площина, k = площина k) і список площин {label, n, d, pixels}.
    """
    rng = np.random.default_rng(seed)
    h, w = valid.shape
    Pf, zf = P.reshape(-1, 3), z.ravel()
    idx = np.flatnonzero(valid.ravel())
    labels = np.zeros(h * w, np.int32)
    min_pts = max(int(min_frac * len(idx)), 3)
    planes = []
    for k in range(1, max_planes + 1):
        if len(idx) < min_pts:
            break
        X = Pf[idx]
        tol = np.maximum(tol_mm, tol_rel * zf[idx])        # допуск росте з відстанню
        # гіпотези: площини через випадкові трійки точок
        tri = rng.integers(0, len(idx), (iters, 3))
        a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm, axis=1)
        ok = ln > 1e-6
        if not ok.any():
            break
        nrm, a = nrm[ok] / ln[ok, None], a[ok]
        d = -(nrm * a).sum(1)
        # оцінка гіпотез на підвибірці
        sub = rng.choice(len(idx), min(5000, len(idx)), replace=False)
        score = (np.abs(X[sub] @ nrm.T + d) < tol[sub, None]).sum(0)
        best = score.argmax()
        inl = np.abs(X @ nrm[best] + d[best]) < tol
        if inl.sum() < min_pts:
            break
        # уточнення через PCA
        for _ in range(2):
            n_, d_ = fit_plane_pca(X[inl])
            inl = np.abs(X @ n_ + d_) < tol
        if inl.sum() < min_pts:
            break
        labels[idx[inl]] = k
        planes.append({"label": k, "n": n_, "d": d_, "pixels": int(inl.sum())})
        idx = idx[~inl]
    return labels.reshape(h, w), planes


# ---------------------------------------------------------------------
# Спершу різати на області, потім RANSAC у кожній
# ---------------------------------------------------------------------


def pixel_normals(P, blur=5):
    """Нормаль кожного пікселя з point map (H, W, 3), у бік камери. blur — згладжування точок."""
    Ps = P.astype(np.float32)
    if blur:
        Ps = cv2.GaussianBlur(Ps, (blur, blur), 0)
    dx = np.zeros_like(Ps)
    dy = np.zeros_like(Ps)
    dx[:, 1:-1] = Ps[:, 2:] - Ps[:, :-2]                 # різниця зліва-направо
    dy[1:-1, :] = Ps[2:, :] - Ps[:-2, :]                 # різниця згори-вниз
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    flip = (n * Ps).sum(-1) > 0                          # нормаль має дивитися на камеру (n·X < 0)
    n[flip] *= -1
    return n


def segment_regions(P, z, valid, angle_deg=10, depth_rel=0.03, blur=5, min_frac=0.005):
    """Ріже кадр на області: межа там, де між сусідами нормаль змінюється більше ніж на
    angle_deg або глибина стрибає більше ніж на depth_rel (частка глибини).
    Повертає маску областей (0 = межа / ненадійне / дрібне) і кількість областей."""
    h, w = z.shape
    n = pixel_normals(P, blur)
    cos_t = np.cos(np.radians(angle_deg))
    cut = ~valid.copy()
    # праві сусіди
    cos_r = (n[:, :-1] * n[:, 1:]).sum(-1)
    jump_r = np.abs(z[:, :-1] - z[:, 1:]) > depth_rel * z[:, :-1]
    edge_r = (cos_r < cos_t) | jump_r
    cut[:, :-1] |= edge_r
    # нижні сусіди
    cos_d = (n[:-1] * n[1:]).sum(-1)
    jump_d = np.abs(z[:-1] - z[1:]) > depth_rel * z[:-1]
    edge_d = (cos_d < cos_t) | jump_d
    cut[:-1] |= edge_d
    keep = (~cut).astype(np.uint8)
    nc, cc, stats, _ = cv2.connectedComponentsWithStats(keep, connectivity=4)
    seg = np.zeros((h, w), np.int32)
    k = 0
    for c in range(1, nc):
        if stats[c, cv2.CC_STAT_AREA] >= min_frac * h * w:
            k += 1
            seg[cc == c] = k
    return seg, k


def segment_then_ransac(P, z, valid, angle_deg=10, depth_rel=0.03, blur=5, seg_min_frac=0.005,
                        inlier_frac=0.3, max_per_region=3, return_segments=False):
    """Спершу області, потім RANSAC у кожній. inlier_frac — площина має містити щонайменше
    цю частку точок області (інакше область неплоска і відкидається)."""
    seg, k = segment_regions(P, z, valid, angle_deg, depth_rel, blur, seg_min_frac)
    lab = np.zeros_like(seg)
    planes = []
    for r in range(1, k + 1):
        m = seg == r
        sub_lab, sub_planes = sequential_ransac(P, z, m, max_planes=max_per_region,
                                                       min_frac=inlier_frac)
        for p in sub_planes:
            new = len(planes) + 1
            lab[sub_lab == p["label"]] = new
            planes.append({**p, "label": new})
    if return_segments:
        return lab, planes, seg
    return lab, planes
