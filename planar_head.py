"""
Етап 3б / Варіант 2. Planar Head поверх VGGT:
    PlanarHead       — легка згорткова голова (DPT-style) поверх ознак VGGT;
    extract_features — витягує проміжні токени VGGT (шари 4, 11, 17, 23) + глибину;
    planar_loss      — комбінований лосс: семантика (CE) + нормалі (Cosine+L1) + ембедінги (Pull-Push);
    find_planes_head — виділення площин за передбаченнями голови (сумісне з pipeline.py та metrics.py);
    cache_features   — один раз проганяє VGGT і зберігає токени + GT кімнати на диск;
    train_head       — навчання голови на кеші (VGGT заморожена), з контролем на val
                       і збереженням найкращих ваг.

Порядок у Colab:
    tr = planes.data.list_rooms(planes.data.load_split("train"))
    va = planes.data.list_rooms(planes.data.load_split("val"))
    planes.planar_head.cache_features(model, tr + va)
    head, hist = planes.planar_head.train_head(tr, va, epochs=20, save_path=".../head.pt")

Передбачення голови для кожного пікселя (розмір VGGT h×w, зазвичай 294×518):
    sem_logits: (N, 4, h, w)  класи з gt.semantic_classes: 0 інше, 1 стіна, 2 підлога, 3 стеля
    normal:     (N, 3, h, w)  одиничні нормалі площин у координатах камери (d > 0)
    embedding:  (N, E, h, w)  ознаки екземплярів для розділення різних стін
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import aggregation, data, gt, metrics, pipeline, ransac

# VGGT-1B має 24 блоки агрегатора, кожен видає токени розмірності 2048 (1024 кадрових + 1024 глобальних)
LAYER_IDS = (4, 11, 17, 23)
PATCH_SIZE = 14
ID_TYPE = {1: "wall", 2: "floor", 3: "ceiling"}


class _ResidualConvUnit(nn.Module):
    """Залишковий згортковий блок для злиття різномасштабних ознак (у стилі DPT)."""

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(features)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(self.act(x))))
        out = self.bn2(self.conv2(out))
        return x + out


class _FeatureFusionBlock(nn.Module):
    """Злиття ознак двох рівнів із підвищенням роздільної здатності вдвічі."""

    def __init__(self, features):
        super().__init__()
        self.res1 = _ResidualConvUnit(features)
        self.res2 = _ResidualConvUnit(features)
        self.proj = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, x, skip=None, out_hw=None):
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.res1(skip)
        x = self.res2(x)
        if out_hw is not None:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class PlanarHead(nn.Module):
    """Голова реконструкції площин поверх токенів агрегатора VGGT та карти глибини.

    Приймає:
        tokens_list — список із 4 тензорів токенів (N, P_patches, token_dim) з шарів LAYER_IDS
        depth       — карта глибини VGGT (N, 1, h, w) для геометричної прив'язки (опційно)
        hw          — цільовий розмір (h, w), кратний PATCH_SIZE=14

    Повертає словник:
        sem_logits  — (N, 4, h, w) логіти класів (0: інше, 1: стіна, 2: підлога, 3: стеля)
        normal      — (N, 3, h, w) одиничні вектори нормалей у системі камери
        embedding   — (N, emb_dim, h, w) L2-нормовані ембедінги екземплярів площин
    """

    def __init__(self, token_dim=2048, features=256, num_classes=4, emb_dim=8, use_depth=True):
        super().__init__()
        self.use_depth = use_depth

        # Проєкція токенів із 4 рівнів трансформера у спільний простір ознак
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(token_dim, features, kernel_size=1, bias=False),
                nn.BatchNorm2d(features),
                nn.GELU(),
            )
            for _ in range(4)
        ])

        # Різнорівневе масштабування (піраміда ознак x4, x2, x1, x1/2 відносно сітки патчів)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1),
        ])

        # Каскадне злиття рівнів (від найглибшого до найбільш детального)
        self.refinenet4 = _FeatureFusionBlock(features)
        self.refinenet3 = _FeatureFusionBlock(features)
        self.refinenet2 = _FeatureFusionBlock(features)
        self.refinenet1 = _FeatureFusionBlock(features)

        # Гілка геометричних ознак із карти глибини та її градієнтів
        geo_ch = 32 if use_depth else 0
        if use_depth:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(3, geo_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(geo_ch),
                nn.GELU(),
                nn.Conv2d(geo_ch, geo_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(geo_ch),
                nn.GELU(),
            )

        in_ch = features + geo_ch

        # Вихідні гілки: семантика (4 класи), нормаль (3 компоненти), ембедінг екземпляра
        self.sem_head = nn.Sequential(
            nn.Conv2d(in_ch, features // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features // 2),
            nn.GELU(),
            nn.Conv2d(features // 2, num_classes, kernel_size=1),
        )
        self.normal_head = nn.Sequential(
            nn.Conv2d(in_ch, features // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features // 2),
            nn.GELU(),
            nn.Conv2d(features // 2, 3, kernel_size=1),
        )
        self.emb_head = nn.Sequential(
            nn.Conv2d(in_ch, features // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features // 2),
            nn.GELU(),
            nn.Conv2d(features // 2, emb_dim, kernel_size=1),
        )

    @staticmethod
    def _depth_clues(depth):
        """Нормована глибина та її градієнти вздовж u, v як підказка про злами площин."""
        d_norm = depth / (depth.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6))
        du = F.pad(d_norm[..., :, 1:] - d_norm[..., :, :-1], (0, 1, 0, 0))
        dv = F.pad(d_norm[..., 1:, :] - d_norm[..., :-1, :], (0, 0, 0, 1))
        return torch.cat([d_norm, du, dv], dim=1)

    def forward(self, tokens_list, hw, depth=None):
        h, w = hw
        ph, pw = h // PATCH_SIZE, w // PATCH_SIZE

        feats = []
        for tok, proj, res in zip(tokens_list, self.projects, self.resize_layers):
            n, _, c = tok.shape
            fmap = tok.transpose(1, 2).reshape(n, c, ph, pw).contiguous()
            feats.append(res(proj(fmap)))

        p4 = self.refinenet4(feats[3], out_hw=feats[2].shape[-2:])
        p3 = self.refinenet3(p4, feats[2], out_hw=feats[1].shape[-2:])
        p2 = self.refinenet2(p3, feats[1], out_hw=feats[0].shape[-2:])
        fused = self.refinenet1(p2, feats[0], out_hw=(h, w))

        if self.use_depth:
            if depth is None:
                geo = torch.zeros(
                    (fused.shape[0], 32, h, w), device=fused.device, dtype=fused.dtype
                )
            else:
                if depth.ndim == 3:
                    depth = depth.unsqueeze(1)
                geo = self.depth_encoder(self._depth_clues(depth))
            fused = torch.cat([fused, geo], dim=1)

        sem_logits = self.sem_head(fused)
        normal = F.normalize(self.normal_head(fused), dim=1, eps=1e-6)
        embedding = F.normalize(self.emb_head(fused), dim=1, eps=1e-6)
        return {"sem_logits": sem_logits, "normal": normal, "embedding": embedding}


def extract_vggt_features(model, names, layer_ids=LAYER_IDS):
    """Проганяє VGGT (без градієнтів) і повертає проміжні токени та геометрію кімнати.

    Повертає словник:
        tokens: список із 4 тензорів (S, P_patches, 2048) на CPU/GPU
        depth:  (S, h, w) numpy
        conf:   (S, h, w) numpy
        extr:   (S, 3, 4) numpy
        intr:   (S, 3, 3) numpy
        hw:     (h, w)
    """
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = next(model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    imgs = load_and_preprocess_images([data.frame_path(n, "rgb") for n in names]).to(device)
    if imgs.ndim == 4:
        imgs = imgs.unsqueeze(0)                              # (1, S, 3, h, w)
    h, w = imgs.shape[-2:]

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        agg_tokens_list, patch_start_idx = model.aggregator(imgs)
        pose_enc = model.camera_head(agg_tokens_list)[-1]
        depth_map, depth_conf = model.depth_head(
            agg_tokens_list, images=imgs, patch_start_idx=patch_start_idx
        )

    ex, ik = pose_encoding_to_extri_intri(pose_enc, (h, w))
    tokens = [
        agg_tokens_list[i][0, :, patch_start_idx:, :].float().cpu()
        for i in layer_ids
    ]
    out = {
        "tokens": tokens,
        "depth": depth_map[0, ..., 0].float().cpu().numpy(),
        "conf": depth_conf[0].float().cpu().numpy(),
        "extr": ex[0].float().cpu().numpy(),
        "intr": ik[0].float().cpu().numpy(),
        "hw": (h, w),
    }
    del agg_tokens_list, imgs, depth_map, depth_conf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Функції втрат (Loss) та підготовка еталонних тензорів із gt.py
# ---------------------------------------------------------------------------

def build_gt_targets(f, hw):
    """Готує еталонні карти для одного кадру у розмірі VGGT (h, w):
        sem_gt:    (h, w) int64   — 0 інше, 1 стіна, 2 підлога, 3 стеля (з gt.semantic_classes)
        normal_gt: (3, h, w) f32  — нормаль площини в координатах камери (d > 0)
        inst_gt:   (h, w) int64   — номер площини (0 = не площина, k = площина k)
    """
    h, w = hw
    sem_full = gt.semantic_classes(f)
    g = gt.gt_frame(f, use_semantic=False)
    inst_full = g["mask"]

    sem_gt = cv2.resize(sem_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    inst_gt = cv2.resize(inst_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)

    normal_gt = np.zeros((3, h, w), dtype=np.float32)
    for p in g["planes"]:
        sel = inst_gt == p["label"]
        if sel.any():
            normal_gt[:, sel] = p["n"][:, None].astype(np.float32)
    return {
        "sem": torch.from_numpy(sem_gt),
        "normal": torch.from_numpy(normal_gt),
        "inst": torch.from_numpy(inst_gt),
    }


def _pull_push_loss(emb, inst, margin_pull=0.1, margin_push=0.5, max_pts=512):
    """Дискримінативний лосс для ембедінгів площин:
        тягне пікселі однієї площини до її середнього вектора (pull),
        відштовхує центри різних площин одного кадру (push)."""
    device = emb.device
    total_pull = torch.tensor(0.0, device=device)
    total_push = torch.tensor(0.0, device=device)
    valid_frames = 0

    for b in range(emb.shape[0]):
        eb = emb[b].permute(1, 2, 0)                         # (h, w, E)
        ib = inst[b]                                         # (h, w)
        labels = [int(l) for l in torch.unique(ib) if l > 0]
        if not labels:
            continue
        valid_frames += 1
        centers = []
        pull_b = torch.tensor(0.0, device=device)
        for l in labels:
            pts = eb[ib == l]
            if len(pts) > max_pts:
                perm = torch.randperm(len(pts), device=device)[:max_pts]
                pts = pts[perm]
            mu = F.normalize(pts.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
            centers.append(mu.squeeze(0))
            dist = torch.norm(pts - mu, dim=-1)
            pull_b = pull_b + F.relu(dist - margin_pull).mean()
        total_pull = total_pull + pull_b / len(labels)

        if len(centers) >= 2:
            C = torch.stack(centers, dim=0)                  # (K, E)
            pdist = torch.cdist(C, C, p=2)
            triu = torch.triu(torch.ones_like(pdist, dtype=torch.bool), diagonal=1)
            push_b = F.relu(2.0 * margin_push - pdist[triu]).mean()
            total_push = total_push + push_b

    if valid_frames == 0:
        return total_pull
    return (total_pull + total_push) / valid_frames


def planar_loss(pred, target, w_sem=1.0, w_norm=1.0, w_emb=0.5):
    """Сумарна функція втрат PlanarHead:
        L = w_sem * CE(sem) + w_norm * (1 - cos + L1)(normal) + w_emb * PullPush(emb)."""
    sem_gt = target["sem"].to(pred["sem_logits"].device)
    norm_gt = target["normal"].to(pred["normal"].device)
    inst_gt = target["inst"].to(pred["embedding"].device)

    loss_sem = F.cross_entropy(pred["sem_logits"], sem_gt)

    planar_mask = (inst_gt > 0) & (norm_gt.norm(dim=1) > 0.5)
    if planar_mask.any():
        n_pred = pred["normal"].permute(0, 2, 3, 1)[planar_mask]
        n_true = norm_gt.permute(0, 2, 3, 1)[planar_mask]
        cos_loss = (1.0 - (n_pred * n_true).sum(dim=-1)).mean()
        l1_loss = F.l1_loss(n_pred, n_true)
        loss_norm = cos_loss + l1_loss
    else:
        loss_norm = torch.tensor(0.0, device=sem_gt.device)

    loss_emb = _pull_push_loss(pred["embedding"], inst_gt)
    total = w_sem * loss_sem + w_norm * loss_norm + w_emb * loss_emb
    return {
        "loss": total,
        "loss_sem": float(loss_sem.detach()),
        "loss_norm": float(loss_norm.detach()),
        "loss_emb": float(loss_emb.detach()),
    }


# ---------------------------------------------------------------------------
# Інференс: виділення площин за виходами голови та інтеграція з pipeline.py
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_frame_maps(head, feat_room, frame_idx=0):
    """Запускає PlanarHead для одного кадру кімнати -> numpy-карти sem_cls, sem_prob, normal, emb."""
    device = next(head.parameters()).device
    tokens_s = [t[frame_idx : frame_idx + 1].to(device) for t in feat_room["tokens"]]
    depth_s = torch.from_numpy(feat_room["depth"][frame_idx : frame_idx + 1]).float().to(device)
    pred = head(tokens_s, feat_room["hw"], depth=depth_s)

    probs = F.softmax(pred["sem_logits"][0], dim=0).cpu().numpy()
    sem_cls = probs.argmax(axis=0).astype(np.uint8)
    sem_prob = probs.max(axis=0)
    normal = pred["normal"][0].permute(1, 2, 0).cpu().numpy()
    emb = pred["embedding"][0].permute(1, 2, 0).cpu().numpy()
    return {"sem_cls": sem_cls, "sem_prob": sem_prob, "normal": normal, "emb": emb}


def _split_walls_by_normal_and_emb(mask_wall, normal_map, emb_map, P, z,
                                   ang_thr=15.0, emb_thr=0.35, min_px=150):
    """Розділяє маску стін (клас 1) на окремі площини за нормалями голови, ембедінгами та RANSAC."""
    sub_labels = np.zeros_like(mask_wall, dtype=np.int32)
    if mask_wall.sum() < min_px:
        return sub_labels, []

    # 1. Спочатку запускаємо послідовний RANSAC виключно на пікселях стін
    lab_r, planes_r = ransac.sequential_ransac(P, z, mask_wall)
    if not planes_r:
        return sub_labels, []

    # 2. Перевіряємо узгодженість із нормалями голови й розбиваємо зв'язні компоненти
    lab_c, planes_c = aggregation.clean_planes(lab_r, planes_r, P)
    if not planes_c:
        return sub_labels, []

    # 3. Зливаємо копланарні шматки однієї стіни, якщо їхні нормалі та ембедінги близькі
    means_emb = []
    means_nrm = []
    for p in planes_c:
        sel = lab_c == p["label"]
        e_mean = emb_map[sel].mean(axis=0)
        e_mean /= max(np.linalg.norm(e_mean), 1e-6)
        n_mean = normal_map[sel].mean(axis=0)
        n_mean /= max(np.linalg.norm(n_mean), 1e-6)
        means_emb.append(e_mean)
        means_nrm.append(n_mean)

    def ok(i, j):
        a, b = planes_c[i], planes_c[j]
        ang_geom = np.degrees(np.arccos(np.clip(a["n"] @ b["n"], -1, 1)))
        ang_head = np.degrees(np.arccos(np.clip(means_nrm[i] @ means_nrm[j], -1, 1)))
        d_diff = abs(a["d"] - b["d"])
        d_lim = max(60.0, 0.03 * (a["d"] + b["d"]) / 2.0)
        emb_dist = np.linalg.norm(means_emb[i] - means_emb[j])
        return (ang_geom < 7.0 or ang_head < ang_thr) and d_diff < d_lim and emb_dist < emb_thr * 2.0

    roots = aggregation._union_find(len(planes_c), ok)
    groups = {}
    for i, r in enumerate(roots):
        groups.setdefault(r, []).append(planes_c[i]["label"])
    return aggregation._relabel(lab_c, groups.values(), P)


def find_planes_head(P, z, valid, head_maps, prob_thr=0.45, min_frac=0.005,
                     boundary=False, max_behind=0.03, behind_max_mm=None):
    """Знаходить площини кадру за допомогою передбачень PlanarHead.

    Семантична маска голови автоматично відсікає меблі та сторонні об'єкти (клас 0),
    розділяє підлогу (клас 2), стелю (клас 3) та окремі стіни (клас 1).
    Повертає (lab, planes) у тому самому форматі, що й pipeline.find_planes.
    """
    h, w = valid.shape
    min_px = int(min_frac * h * w)
    sem_cls = head_maps["sem_cls"]
    sem_prob = head_maps["sem_prob"]
    normal_map = head_maps["normal"]
    emb_map = head_maps["emb"]

    usable = valid & (sem_cls > 0) & (sem_prob >= prob_thr)
    out_lab = np.zeros((h, w), dtype=np.int32)
    planes = []

    # Стіни (клас 1): може бути декілька різних площин у кадрі
    wall_mask = usable & (sem_cls == 1)
    w_lab, w_planes = _split_walls_by_normal_and_emb(
        wall_mask, normal_map, emb_map, P, z, min_px=max(min_px, 50)
    )
    for p in w_planes:
        sel = w_lab == p["label"]
        if sel.sum() < min_px:
            continue
        k = len(planes) + 1
        out_lab[sel] = k
        planes.append({**p, "label": k, "type": "wall"})

    # Підлога (клас 2) та стеля (клас 3): зазвичай по одній головній площині на клас
    for cls_id in (2, 3):
        cls_mask = usable & (sem_cls == cls_id)
        if cls_mask.sum() < min_px:
            continue
        lab_c, pl_c = ransac.sequential_ransac(P, z, cls_mask, max_planes=3)
        if not pl_c:
            continue
        lab_c, pl_c = aggregation.clean_planes(lab_c, pl_c, P, min_frac=min_frac)
        if not pl_c:
            continue
        lab_c, pl_c = aggregation.merge_coplanar(lab_c, pl_c, P)
        for p in pl_c:
            sel = lab_c == p["label"]
            if sel.sum() < min_px:
                continue
            k = len(planes) + 1
            out_lab[sel] = k
            planes.append({**p, "label": k, "type": ID_TYPE[cls_id]})

    if boundary and planes:
        out_lab, planes = aggregation.filter_boundary(
            out_lab, planes, P, valid, max_behind=max_behind, behind_max_mm=behind_max_mm
        )
    return out_lab, planes


def process_room_head(room, model, head, conf_pct=30, prob_thr=0.45, boundary=False,
                      max_behind=0.03, behind_max_mm=None, scale_mode="self",
                      gt_semantic=False, return_planes=False):
    """Повний пайплайн кімнати з PlanarHead (аналог pipeline.process_room для варіанта 2)."""
    names = data.room_frames(room)
    feat = extract_vggt_features(model, names)
    loaded = [data.load_frame(n) for n in names]
    sc, eval_fix = pipeline._scales(feat, loaded, scale_mode, conf_pct)

    head.eval()
    rows, items, frames_out = [], [], []
    for s, f in enumerate(loaded):
        z = feat["depth"][s] * sc
        P = data.backproject(z, feat["intr"][s])
        valid = feat["conf"][s] > np.percentile(feat["conf"][s], conf_pct)
        head_maps = predict_frame_maps(head, feat, frame_idx=s)

        lab, planes = find_planes_head(
            P, z, valid, head_maps, prob_thr=prob_thr, boundary=boundary,
            max_behind=max_behind, behind_max_mm=behind_max_mm
        )

        H, W = f["rgb"].shape[:2]
        lab_full = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
        g = gt.gt_frame(f, use_semantic=gt_semantic)
        planes_eval = [{**p, "d": p["d"] * eval_fix} for p in planes]
        m = metrics.evaluate(lab_full, planes_eval, g)
        gt_id = {q["label"]: q["id"] for q in g["planes"]}
        match = {a: gt_id[b] for a, b, _, _ in m.pop("pairs")}
        rows.append({"кадр": f["name"], **m})

        R, t = feat["extr"][s][:, :3], feat["extr"][s][:, 3] * sc
        for it in aggregation.to_world(planes, lab, P, R, t):
            items.append({**it, "frame": s, "gt_id": match.get(it["label"])})
        if return_planes:
            frames_out.append({
                "name": f["name"], "rgb": f["rgb"], "gt": g["mask"],
                "lab_full": lab_full, "planes": planes, "sem_cls": head_maps["sem_cls"],
            })

    gid = aggregation.merge_items(items)
    merge = {
        "кімната": room, "кадрів": len(names), "масштаб_помилка": round(eval_fix, 3),
        "площин_до": len(items), "площин_після": max(gid) if gid else 0,
        **metrics.merge_consistency(items, gid),
    }

    if return_planes:
        for s, fo in enumerate(frames_out):
            lut = np.zeros(int(fo["lab_full"].max()) + 1, np.int32)
            for i, it in enumerate(items):
                if it["frame"] == s:
                    lut[it["label"]] = gid[i]
            fo["scene_lab"] = lut[fo["lab_full"]]
        return rows, merge, frames_out
    return rows, merge


def run_all_head(model, head, rooms=None, scenes=None, **kw):
    """Запуск пайплайну з PlanarHead по списку кімнат (аналог pipeline.run_all)."""
    rooms = rooms or data.list_rooms(scenes)
    rows, merges = [], []
    for room in rooms:
        r, m = process_room_head(room, model, head, **kw)
        rows += r
        merges.append(m)
    summ = metrics.summarize(rows)
    mg = pd.DataFrame(merges) if merges else None
    if mg is not None and not mg.empty:
        summ["злито_правильно"] = f'{mg["злито_правильно"].sum()}/{mg["стін_на_2+_кадрах"].sum()}'
        summ["злито_різні"] = int(mg["злито_різні"].sum())
    return summ, pd.DataFrame(rows), mg


# ---------------------------------------------------------------------------
# Навчання та збереження / завантаження ваг PlanarHead
# ---------------------------------------------------------------------------

CACHE_DIR = "/content/cache"     # локальний диск Colab: швидкий, але зникає після перезапуску


def _cache_path(room, cache_dir=None):
    return os.path.join(cache_dir or CACHE_DIR, room + ".pt")


def cache_features(model, rooms, cache_dir=None):
    """Один раз проганяє VGGT для кожної кімнати і зберігає токени (fp16), глибину та GT.
    Уже збережені кімнати пропускає, тож після обриву можна просто запустити знову.
    Розмір: ~25–40 МБ на кімнату."""
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    todo = [r for r in rooms if not os.path.exists(_cache_path(r, cache_dir))]
    print(f"кеш: {len(rooms) - len(todo)} уже є, рахую {len(todo)}")
    for i, room in enumerate(todo, 1):
        names = data.room_frames(room)
        if not names:
            continue
        feat = extract_vggt_features(model, names)
        feat["tokens"] = [t.half() for t in feat["tokens"]]           # вдвічі менше місця
        targets = [build_gt_targets(data.load_frame(n), feat["hw"]) for n in names]
        torch.save({"feat": feat, "targets": targets}, _cache_path(room, cache_dir))
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")


def _load_cached(room, device, cache_dir=None):
    """Кімната з кешу -> токени, глибина, розмір, GT-тензори (усе на device)."""
    c = torch.load(_cache_path(room, cache_dir), map_location="cpu", weights_only=False)
    feat, targets = c["feat"], c["targets"]
    tokens = [t.float().to(device) for t in feat["tokens"]]
    depth = torch.from_numpy(feat["depth"]).float().to(device)
    target = {k: torch.stack([t[k] for t in targets]).to(device) for k in ("sem", "normal", "inst")}
    return tokens, depth, feat["hw"], target


def _run_epoch(head, rooms, device, opt=None, cache_dir=None):
    """Один прохід по кімнатах. opt=None — лише оцінка (val), без навчання."""
    train = opt is not None
    head.train(train)
    out = []
    for room in rooms:
        if not os.path.exists(_cache_path(room, cache_dir)):
            continue
        tokens, depth, hw, target = _load_cached(room, device, cache_dir)
        with torch.set_grad_enabled(train):
            pred = head(tokens, hw, depth=depth)
            losses = planar_loss(pred, target)
        if train:
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        out.append({k: float(v) for k, v in losses.items()})
    if not out:
        return {}
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def train_head(train_rooms, val_rooms=None, epochs=20, lr=1e-3, weight_decay=1e-4,
               save_path=None, device="cuda", cache_dir=None, seed=0):
    """Навчає PlanarHead на закешованих кімнатах (спершу cache_features).

    Після кожної епохи — loss на val; save_path зберігає ваги з НАЙМЕНШИМ val loss
    (якщо val немає — ваги останньої епохи).
    Перевірка, що код учиться: train_head([room], epochs=200) — loss має впасти майже до 0."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = PlanarHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history, best = [], float("inf")
    train_rooms = list(train_rooms)

    for ep in range(1, epochs + 1):
        rng.shuffle(train_rooms)
        tr = _run_epoch(head, train_rooms, device, opt, cache_dir)
        sched.step()
        row = {"epoch": ep, **{"train_" + k: round(v, 4) for k, v in tr.items()}}
        msg = f"епоха {ep}/{epochs}: train loss={tr.get('loss', 0):.4f}"
        if val_rooms:
            va = _run_epoch(head, val_rooms, device, None, cache_dir)
            row.update({"val_" + k: round(v, 4) for k, v in va.items()})
            msg += f"  val loss={va.get('loss', 0):.4f}"
            score = va.get("loss", float("inf"))
        else:
            score = -ep                                          # без val — остання епоха
        if save_path and score < best:
            best = score
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(head.state_dict(), save_path)
            msg += "  <- збережено"
        history.append(row)
        print(msg)

    if save_path and os.path.exists(save_path):
        head.load_state_dict(torch.load(save_path, map_location=device))   # повертаємо найкращу
    return head, pd.DataFrame(history)


def show_prediction(head, model, room, frame_idx=0):
    """Фото / GT-класи / передбачені класи для одного кадру (0 інше, 1 стіна, 2 підлога, 3 стеля)."""
    import matplotlib.pyplot as plt
    names = data.room_frames(room)
    feat = extract_vggt_features(model, names)
    head.eval()
    maps = predict_frame_maps(head, feat, frame_idx)
    f = data.load_frame(names[frame_idx])
    h, w = feat["hw"]
    sem_gt = cv2.resize(gt.semantic_classes(f), (w, h), interpolation=cv2.INTER_NEAREST)
    pal = np.array([[0, 0, 0], [174, 199, 232], [152, 223, 138], [78, 71, 183]], np.uint8)
    acc = (maps["sem_cls"] == sem_gt).mean()
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for a, (title, img) in zip(ax, [("фото", cv2.resize(f["rgb"], (w, h))),
                                    ("GT-класи", pal[sem_gt]),
                                    (f"голова (збіг {acc:.0%})", pal[maps["sem_cls"]]),
                                    ("нормалі голови", ((maps["normal"] + 1) / 2 * 255).astype(np.uint8))]):
        a.imshow(img)
        a.set_title(title)
        a.axis("off")
    plt.show()


def load_head(ckpt_path=None, device="cuda", **kw):
    """Створює PlanarHead і (за наявності) завантажує збережені ваги."""
    head = PlanarHead(**kw).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        head.load_state_dict(torch.load(ckpt_path, map_location=device))
    head.eval()
    return head