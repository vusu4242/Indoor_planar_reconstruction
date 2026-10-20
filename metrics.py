"""
Етап 6. Оцінювання.

Протокол:
    - зіставлення площин з GT угорським алгоритмом за IoU (поріг 0.5);
    - IoU рахується лише по пікселях, які GT вважає площинами
      (дзеркала, двері в площині стіни тощо не карають);
    - GT-площини менші за 0.5 % кадру не оцінюються;
    - 3D-похибки (кут нормалі, відстань) — лише для зіставлених пар.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def evaluate(lab_full, planes, g, iou_thr=0.5, min_frac=0.005):
    """Метрики одного кадру. lab_full — маска передбачених площин у розмірі GT."""
    gm = g["mask"]
    known = gm > 0
    min_px = min_frac * gm.size
    G = [q for q in g["planes"] if (gm == q["label"]).sum() >= min_px]

    iou = np.zeros((len(planes), len(G)))
    for i, p in enumerate(planes):
        pm = (lab_full == p["label"]) & known
        for j, q in enumerate(G):
            qm = gm == q["label"]
            inter = (pm & qm).sum()
            if inter:
                iou[i, j] = inter / (pm | qm).sum()

    r, c = linear_sum_assignment(-iou) if iou.size else ([], [])
    pairs = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_thr]
    tp, fp, fn = len(pairs), len(planes) - len(pairs), len(G) - len(pairs)

    angs = [np.degrees(np.arccos(np.clip(abs(planes[i]["n"] @ G[j]["n"]), 0, 1))) for i, j in pairs]
    offs = [abs(planes[i]["d"] - G[j]["d"]) for i, j in pairs]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec,
            "F1": 2 * prec * rec / max(prec + rec, 1e-9),
            "mIoU": float(np.mean([iou[i, j] for i, j in pairs])) if pairs else 0.0,
            "angle_deg": float(np.mean(angs)) if angs else np.nan,
            "offset_mm": float(np.mean(offs)) if offs else np.nan,
            "pairs": [(planes[i]["label"], G[j]["label"], G[j]["type"], iou[i, j]) for i, j in pairs]}


def summarize(rows):
    """Підсумок по всіх кадрах: precision/recall/F1 із сумарних TP/FP/FN."""
    df = pd.DataFrame(rows)
    tp, fp, fn = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": round(p, 2), "recall": round(r, 2),
            "F1": round(2 * p * r / max(p + r, 1e-9), 2),
            "mIoU": round(df.loc[df["TP"] > 0, "mIoU"].mean(), 2),
            "angle_deg": round(df["angle_deg"].mean(), 2),
            "offset_mm": round(df["offset_mm"].mean(), 1)}


def merge_consistency(items, gid):
    """Якість злиття між кадрами за GT-ID стін.
    items[i]["gt_id"] — ID GT-стіни, з якою зіставлено площину (або None); gid[i] — її група."""
    seen, groups = {}, {}
    for i, it in enumerate(items):
        if it.get("gt_id") is not None:
            seen.setdefault(it["gt_id"], []).append(gid[i])
            groups.setdefault(gid[i], set()).add(it["gt_id"])
    multi = {k: v for k, v in seen.items() if len(v) >= 2}
    return {"стін_на_2+_кадрах": len(multi),
            "злито_правильно": sum(len(set(v)) == 1 for v in multi.values()),
            "злито_різні": sum(len(v) > 1 for v in groups.values())}
