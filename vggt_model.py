"""
Етап 3. Foundation-модель VGGT: фото кімнати -> глибина, впевненість, камери.

Встановлення в Colab (потім Runtime -> Restart session):
    !pip install -q "numpy>=2.1,<2.3"
    !pip install -q --no-deps git+https://github.com/facebookresearch/vggt.git
    !pip install -q huggingface_hub safetensors einops
"""
from . import data


def load_model():
    """Завантажує VGGT-1B на GPU."""
    from vggt.models.vggt import VGGT
    return VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()


def run_vggt(model, names):
    """VGGT на кадрах однієї кімнати. Усі виходи — numpy, розмір VGGT (зазвичай 294×518).

    depth: (кадри, h, w)   глибина в умовному масштабі
    conf:  (кадри, h, w)   впевненість у глибині
    extr:  (кадри, 3, 4)   камера: X_cam = R·X_world + t (світ = перша камера)
    intr:  (кадри, 3, 3)   матриця K для розміру h×w
    """
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    imgs = load_and_preprocess_images([data.frame_path(n, "rgb") for n in names]).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        pr = model(imgs)
    ex, ik = pose_encoding_to_extri_intri(pr["pose_enc"], imgs.shape[-2:])
    out = {"depth": pr["depth"][0, ..., 0].float().cpu().numpy(),
           "conf": pr["depth_conf"][0].float().cpu().numpy(),
           "extr": ex[0].float().cpu().numpy(),
           "intr": ik[0].float().cpu().numpy()}
    del pr, imgs
    torch.cuda.empty_cache()
    return out
