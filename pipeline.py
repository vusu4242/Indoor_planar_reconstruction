"""
(7) VGGT + Aggregation pipeline for a room:
    VGGT -> scale -> RANSAC -> cleaning -> merging pieces -> boundary filter
    -> merging planes across frames -> evaluation

scale_mode="oracle" takes the scale from GT before plane detection: an upper bound,
not an honest result.
"""
import cv2
import numpy as np
import pandas as pd

from . import aggregation, data, gt, metrics, ransac, vggt_model

_VGGT_CACHE = {}          # VGGT outputs per room, so the model is not run twice for the same room


def get_vggt(room, model, use_cache=True):
    """Gets VGGT outputs for a room (depth, conf, extr, intr); taken from the cache if already computed."""
    if use_cache and room in _VGGT_CACHE:
        return _VGGT_CACHE[room]
    v = vggt_model.run_vggt(model, data.room_frames(room))
    if use_cache:
        _VGGT_CACHE[room] = v
    return v


def self_scale(depth_pred, conf, conf_pct=30, target_mm=2500.0):
    """Scales from arbitrary measure to mm
    Parameters:
    target_mm - the median depth of reliable pixels in the room(a typical distance in a room)
    Returns the factor that converts VGGT depth to mm"""
    vals = [depth_pred[s][conf[s] > np.percentile(conf[s], conf_pct)] for s in range(len(depth_pred))]
    return float(target_mm / np.median(np.concatenate(vals)))


def align_scale(depth_pred, loaded):
    """Scale FROM GT (oracle): median ratio of GT depth to VGGT depth.
    Uses the ground truth, so only for evaluation or as an upper bound."""
    h, w = depth_pred.shape[1:]
    scales = []
    for s, f in enumerate(loaded):
        gs = cv2.resize(f["depth"].astype(np.float64), (w, h), interpolation=cv2.INTER_NEAREST)
        ok = gs > 0
        scales.append(np.median(gs[ok] / depth_pred[s][ok]))
    return float(np.median(scales))


def find_planes(P, z, valid, inframe_merge=True, boundary=True, max_behind=0.03,
                behind_max_mm=None, open_k=3, close_k=0, footprint_margin_mm=None,
                ransac_min_frac=0.01, clean_min_frac=0.005, return_stages=False):
    """Planes of one frame
    Inputs:
    P, z, valid:   3D points (H, W, 3) in mm, depth (H, W) in mm, pixels to use (H, W)
    Parameters:
    inframe_merge: merge pieces of the same plane (merge_coplanar)
    boundary:      keep only room boundaries (filter_boundary; max_behind, behind_max_mm,
                   footprint_margin_mm are its parameters)
    open_k:        opening kernel in cleaning (0 = off)
    close_k:       closing kernel at the end, fills holes in masks only (0 = off)
    ransac_min_frac: minimal plane size in RANSAC (fraction of valid pixels)
    clean_min_frac:  minimal piece size in cleaning (fraction of the frame)
    return_stages: also return the mask after each stage (for debugging)

    Output:
    (labels (H, W), planes) or (labels, planes, stages)
    """
    stages = []
    lab, planes = ransac.sequential_ransac(P, z, valid, min_frac=ransac_min_frac)
    stages.append(("RANSAC", lab, len(planes)))
    lab, planes = aggregation.clean_planes(lab, planes, P, min_frac=clean_min_frac, open_k=open_k)
    stages.append(("cleaning", lab, len(planes)))
    if inframe_merge:
        lab, planes = aggregation.merge_coplanar(lab, planes, P)
        stages.append(("merging pieces", lab, len(planes)))
    if boundary:
        lab, planes = aggregation.filter_boundary(lab, planes, P, valid, max_behind=max_behind,
                                                  behind_max_mm=behind_max_mm,
                                                  footprint_margin_mm=footprint_margin_mm)
        stages.append(("boundary filter", lab, len(planes)))
    if close_k:
        lab = aggregation.fill_holes(lab, close_k)           # masks only, plane geometry unchanged
        stages.append(("closing", lab, len(planes)))
    if return_stages:
        return lab, planes, stages
    return lab, planes


def _scales(v, loaded, scale_mode, conf_pct):
    """Returns (sc, eval_fix): sc = scale used by the method;
    eval_fix = gt_scale / sc, applied only to plane offsets during evaluation."""
    gt_scale = align_scale(v["depth"], loaded)             # needed only for evaluation
    if scale_mode == "oracle":
        sc = gt_scale
    elif scale_mode == "self":
        sc = self_scale(v["depth"], v["conf"], conf_pct)
    else:
        raise ValueError("scale_mode: 'self' or 'oracle'")
    return sc, gt_scale / sc


def debug_frame(room, model, frame_idx=0, conf_pct=30, scale_mode="self", gt_semantic=False,
                bilateral_d=0, bilateral_sigma_rel=0.02, **kw):
    """Masks of one frame after each stage, to see where planes get lost.
    bilateral_d, bilateral_sigma_rel: optional depth smoothing (0 = off); **kw go to find_planes.
    Returns (rgb, GT mask, [(stage name, mask at full photo size, number of planes), ...])."""
    names = data.room_frames(room)
    v = get_vggt(room, model)
    loaded = [data.load_frame(n) for n in names]
    sc, _ = _scales(v, loaded, scale_mode, conf_pct)
    s, f = frame_idx, loaded[frame_idx]
    z = data.smooth_depth(v["depth"][s] * sc, bilateral_d, bilateral_sigma_rel)
    P = data.backproject(z, v["intr"][s])
    valid = v["conf"][s] > np.percentile(v["conf"][s], conf_pct)
    _, _, stages = find_planes(P, z, valid, return_stages=True, **kw)
    H, W = f["rgb"].shape[:2]
    up = lambda m: cv2.resize(m.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    out = [("pixels after confidence filter", up(valid), int(valid.mean() * 100))]
    out += [(name, up(lab), n) for name, lab, n in stages]
    return f["rgb"], gt.gt_frame(f, use_semantic=gt_semantic)["mask"], out


def process_room(room, model, conf_pct=30, inframe_merge=True, boundary=True,
                 max_behind=0.03, behind_max_mm=None, open_k=3, close_k=0, footprint_margin_mm=None,
                 ransac_min_frac=0.01, clean_min_frac=0.005, bilateral_d=0, bilateral_sigma_rel=0.02,
                 use_cache=True, return_planes=False, scale_mode="self", gt_semantic=False):
    """Full pipeline for one room: planes in every frame, evaluation, merging across frames.
    Parameters:
    conf_pct:   drop this percentage of least confident pixels in every frame
    scale_mode:
        "self"   - the method does not see GT; scale from VGGT itself; GT scale is applied
                   only to plane offsets during evaluation (honest mode);
        "oracle" - scale from GT before plane detection (upper bound).
    gt_semantic: GT protocol; True = also require semantics (no windows, blinds, doors).
    return_planes: also return masks and planes of every frame (for visualization).
    Other parameters: see find_planes and data.smooth_depth.

    Output:
    (rows, merge) or (rows, merge, frames):
    rows  - metrics per frame, merge - statistics of merging across frames
    """
    names = data.room_frames(room)
    v = get_vggt(room, model, use_cache)
    loaded = [data.load_frame(n) for n in names]
    sc, eval_fix = _scales(v, loaded, scale_mode, conf_pct)

    rows, items, frames_out = [], [], []
    for s, f in enumerate(loaded):
        # planes of the frame
        z = data.smooth_depth(v["depth"][s] * sc, bilateral_d, bilateral_sigma_rel)
        P = data.backproject(z, v["intr"][s])
        valid = v["conf"][s] > np.percentile(v["conf"][s], conf_pct)
        lab, planes = find_planes(P, z, valid, inframe_merge, boundary, max_behind, behind_max_mm,
                                   open_k, close_k, footprint_margin_mm,
                                   ransac_min_frac, clean_min_frac)

        # evaluation against GT
        H, W = f["rgb"].shape[:2]
        lab_full = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
        g = gt.gt_frame(f, use_semantic=gt_semantic)
        planes_eval = [{**p, "d": p["d"] * eval_fix} for p in planes]   # GT touches only evaluation
        m = metrics.evaluate(lab_full, planes_eval, g)
        gt_id = {q["label"]: q["id"] for q in g["planes"]}
        match = {a: gt_id[b] for a, b, _, _ in m.pop("pairs")}
        rows.append({"кадр": f["name"], **m})

        # planes in world coordinates, for merging across frames
        R, t = v["extr"][s][:, :3], v["extr"][s][:, 3] * sc
        for it in aggregation.to_world(planes, lab, P, R, t):
            items.append({**it, "frame": s, "gt_id": match.get(it["label"])})
        if return_planes:
            frames_out.append({"name": f["name"], "rgb": f["rgb"], "gt": g["mask"],
                               "lab_full": lab_full, "planes": planes})

    gid = aggregation.merge_items(items)
    merge = {"кімната": room, "кадрів": len(names), "масштаб_помилка": round(eval_fix, 3),
             "площин_до": len(items), "площин_після": max(gid) if gid else 0,
             **metrics.merge_consistency(items, gid)}

    if return_planes:
        for s, fo in enumerate(frames_out):
            lut = np.zeros(int(fo["lab_full"].max()) + 1, np.int32)
            for i, it in enumerate(items):
                if it["frame"] == s:
                    lut[it["label"]] = gid[i]
            fo["scene_lab"] = lut[fo["lab_full"]]       # room-level plane ids (after merging)
        return rows, merge, frames_out
    return rows, merge


def run_all(model, rooms=None, scenes=None, **kw):
    """Pipeline over many rooms -> (summary, table per frame, table of merging per room).
    rooms: list of rooms; or scenes: list of scenes; if neither is given, all rooms.
    **kw go to process_room."""
    rooms = rooms or data.list_rooms(scenes)
    rows, merges = [], []
    for room in rooms:
        r, m = process_room(room, model, **kw)
        rows += r
        merges.append(m)
    summ = metrics.summarize(rows)
    mg = pd.DataFrame(merges)
    summ["злито_правильно"] = f'{mg["злито_правильно"].sum()}/{mg["стін_на_2+_кадрах"].sum()}'
    summ["злито_різні"] = int(mg["злито_різні"].sum())
    return summ, pd.DataFrame(rows), mg
