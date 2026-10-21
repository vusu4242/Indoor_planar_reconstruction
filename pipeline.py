"""
Етап 7. Увесь варіант 1 для кімнати:
    VGGT -> масштаб -> RANSAC -> очищення -> злиття шматків -> фільтр меж
    -> злиття площин між кадрами -> оцінка.

GT використовується лише в оцінці (scale_mode="self", за замовчуванням).
Режим scale_mode="oracle" бере масштаб з GT ще до пошуку площин — це верхня межа,
а не чесний результат.
"""
import cv2
import numpy as np
import pandas as pd

from . import aggregation, data, gt, metrics, ransac, vggt_model

_VGGT_CACHE = {}          # результати VGGT по кімнатах, щоб не проганяти модель повторно


def get_vggt(room, model, use_cache=True):
    if use_cache and room in _VGGT_CACHE:
        return _VGGT_CACHE[room]
    v = vggt_model.run_vggt(model, data.room_frames(room))
    if use_cache:
        _VGGT_CACHE[room] = v
    return v


def self_scale(depth_pred, conf, conf_pct=10, target_mm=2500.0):
    """Масштаб БЕЗ GT: медіанна глибина кімнати стає target_mm (типова відстань у кімнаті)."""
    vals = [depth_pred[s][conf[s] > np.percentile(conf[s], conf_pct)] for s in range(len(depth_pred))]
    return float(target_mm / np.median(np.concatenate(vals)))


def align_scale(depth_pred, loaded):
    """Масштаб ЗА GT (оракул): медіана відношення GT-глибини до глибини VGGT.
    Використовує правильну відповідь — лише для оцінки або як верхня межа."""
    h, w = depth_pred.shape[1:]
    scales = []
    for s, f in enumerate(loaded):
        gs = cv2.resize(f["depth"].astype(np.float64), (w, h), interpolation=cv2.INTER_NEAREST)
        ok = gs > 0
        scales.append(np.median(gs[ok] / depth_pred[s][ok]))
    return float(np.median(scales))


def find_planes(P, z, valid, inframe_merge=True, boundary=True, max_behind=0.03,
                behind_max_mm=None, return_stages=False):
    """Площини одного кадру (усе, що не залежить від GT).
    return_stages=True — ще й маски після кожного етапу (для налагодження)."""
    stages = []
    lab, planes = ransac.sequential_ransac(P, z, valid)
    stages.append(("RANSAC", lab, len(planes)))
    lab, planes = aggregation.clean_planes(lab, planes, P)
    stages.append(("очищення", lab, len(planes)))
    if inframe_merge:
        lab, planes = aggregation.merge_coplanar(lab, planes, P)
        stages.append(("злиття шматків", lab, len(planes)))
    if boundary:
        lab, planes = aggregation.filter_boundary(lab, planes, P, valid, max_behind=max_behind,
                                                  behind_max_mm=behind_max_mm)
        stages.append(("фільтр меж", lab, len(planes)))
    if return_stages:
        return lab, planes, stages
    return lab, planes


def _scales(v, loaded, scale_mode, conf_pct):
    """Масштаб для методу і поправка відстаней для оцінки."""
    gt_scale = align_scale(v["depth"], loaded)             # потрібен лише для оцінки
    if scale_mode == "oracle":
        sc = gt_scale
    elif scale_mode == "self":
        sc = self_scale(v["depth"], v["conf"], conf_pct)
    else:
        raise ValueError("scale_mode: 'self' або 'oracle'")
    return sc, gt_scale / sc


def debug_frame(room, model, frame_idx=0, conf_pct=30, scale_mode="self", gt_semantic=False, **kw):
    """Маски одного кадру після кожного етапу, щоб побачити, де губляться площини.
    Повертає (rgb, GT-маска, [(назва етапу, маска у повному розмірі, кількість площин), ...])."""
    names = data.room_frames(room)
    v = get_vggt(room, model)
    loaded = [data.load_frame(n) for n in names]
    sc, _ = _scales(v, loaded, scale_mode, conf_pct)
    s, f = frame_idx, loaded[frame_idx]
    z = v["depth"][s] * sc
    P = data.backproject(z, v["intr"][s])
    valid = v["conf"][s] > np.percentile(v["conf"][s], conf_pct)
    _, _, stages = find_planes(P, z, valid, return_stages=True, **kw)
    H, W = f["rgb"].shape[:2]
    up = lambda m: cv2.resize(m.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    out = [("пікселі після фільтра впевненості", up(valid), int(valid.mean() * 100))]
    out += [(name, up(lab), n) for name, lab, n in stages]
    return f["rgb"], gt.gt_frame(f, use_semantic=gt_semantic)["mask"], out


def process_room(room, model, conf_pct=30, inframe_merge=True, boundary=True,
                 max_behind=0.03, behind_max_mm=None, use_cache=True, return_planes=False,
                 scale_mode="self", gt_semantic=False):
    """scale_mode:
        "self"   — метод не бачить GT; масштаб з самої VGGT; GT-масштаб застосовується
                   лише до відстаней площин під час оцінки (чесний варіант);
        "oracle" — масштаб за GT ще до пошуку площин (як було раніше; верхня межа).
    gt_semantic — протокол GT: True = ще й семантика (без вікон, жалюзі, дверей)."""
    names = data.room_frames(room)
    v = get_vggt(room, model, use_cache)
    loaded = [data.load_frame(n) for n in names]
    sc, eval_fix = _scales(v, loaded, scale_mode, conf_pct)

    rows, items, frames_out = [], [], []
    for s, f in enumerate(loaded):
        z = v["depth"][s] * sc
        P = data.backproject(z, v["intr"][s])
        valid = v["conf"][s] > np.percentile(v["conf"][s], conf_pct)
        lab, planes = find_planes(P, z, valid, inframe_merge, boundary, max_behind, behind_max_mm)

        H, W = f["rgb"].shape[:2]
        lab_full = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
        g = gt.gt_frame(f, use_semantic=gt_semantic)
        planes_eval = [{**p, "d": p["d"] * eval_fix} for p in planes]   # GT торкається лише оцінки
        m = metrics.evaluate(lab_full, planes_eval, g)
        gt_id = {q["label"]: q["id"] for q in g["planes"]}
        match = {a: gt_id[b] for a, b, _, _ in m.pop("pairs")}
        rows.append({"кадр": f["name"], **m})

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
            fo["scene_lab"] = lut[fo["lab_full"]]       # номери площин сцени (після злиття)
        return rows, merge, frames_out
    return rows, merge


def run_all(model, rooms=None, scenes=None, **kw):
    """Пайплайн по кімнатах -> (підсумок, таблиця по кадрах, таблиця злиття).
    rooms — список кімнат; або scenes — список сцен; якщо нічого не задано — усі кімнати."""
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
