"""
Етап 4. Пошук площин у хмарі точок: послідовний RANSAC + уточнення через PCA.
"""
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
