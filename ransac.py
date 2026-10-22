"""
(4) Plane detection in a point cloud:
    fit_plane_pca- best-fit plane through a set of points;
    sequential_ransac- finds planes one by one (find a plane -> remove its points -> repeat).
"""
import numpy as np


def fit_plane_pca(X):
    """Best-fit plane through points X (N, 3): n·X + d = 0.
 
    The plane passes through the mean of the points; its normal is the direction
    of the smallest spread (third singular vector of SVD).
    The normal is oriented towards the camera (d > 0), so planes are comparable."""
    c = X.mean(0)                                          # center of the points
    n = np.linalg.svd(X - c, full_matrices=False)[2][2]    # direction of smallest spread = normal
    d = -n @ c                                             # plane passes through the center
    return (-n, -d) if d < 0 else (n, d)



 
def sequential_ransac(P, z, valid, max_planes=12, iters=300, min_frac=0.01,
                      tol_mm=30, tol_rel=0.02, seed=0):
    """
    Finds planes one by one: find the best plane -> remove its points -> search again
 
    P:          (H, W, 3) 3D points in mm (camera coordinates)
    z:          (H, W) depth in mm
    valid:      (H, W) bool, pixels to use (after the confidence filter)
    max_planes: maximum number of planes per frame
    iters:      number of random plane hypotheses per plane
    min_frac:   a plane must contain at least this fraction of valid pixels
    tol_mm, tol_rel: a point lies on a plane if its distance < max(tol_mm, tol_rel * depth)
    seed:       random seed (results are reproducible)
 
    Outputs:
        labels: (H, W) int mask, 0 = no plane, k = plane k
        planes: list of {label, n, d, pixels}
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
        tol = np.maximum(tol_mm, tol_rel * zf[idx])       
        # hypotheses: planes through random triples of points
        tri = rng.integers(0, len(idx), (iters, 3))
        a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm, axis=1)
        ok = ln > 1e-6                                     # skip triples lying on one line
        if not ok.any():
            break
        nrm, a = nrm[ok] / ln[ok, None], a[ok]
        d = -(nrm * a).sum(1)
        # score hypotheses on a random subsample (faster than on all points)
        sub = rng.choice(len(idx), min(5000, len(idx)), replace=False)
        score = (np.abs(X[sub] @ nrm.T + d) < tol[sub, None]).sum(0)
        best = score.argmax()
        inl = np.abs(X @ nrm[best] + d[best]) < tol        # inliers of the best hypothesis
        if inl.sum() < min_pts:
            break
        # PCA refinement: refit the plane on all inliers, recompute inliers (twice)
        for _ in range(2):
            n_, d_ = fit_plane_pca(X[inl])
            inl = np.abs(X @ n_ + d_) < tol
        if inl.sum() < min_pts:
            break
        labels[idx[inl]] = k
        planes.append({"label": k, "n": n_, "d": d_, "pixels": int(inl.sum())})
        idx = idx[~inl]                                    # remove points of this plane
    return labels.reshape(h, w), planes
 
