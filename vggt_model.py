"""
(3) Foundation model VGGT:
        Input: photo of room(path to dataset)
        Output: depth mask, confidence matrix, cameras(int,ext)
"""
from . import data


def load_model():
    """Downloading VGGT-1B on GPU"""
    from vggt.models.vggt import VGGT
    return VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()


def run_vggt(model, names):
    """VGGT on frames of one room. All outputs are numpy arrays.
   Images are resized before VGGT: width 518, height rounded to a multiple of 14
    (1280×720 -> 518×294, width × height), so depth and conf have shape (294, 518),
    not the size of the original photo.
        Outputs:
        depth: (frames, h, w)   depth in arbitrary scale (not mm)
        conf:  (frames, h, w)   confidence of depth (>= 1)
        extr:  (frames, 3, 4)   camera: X_cam = R·X_world + t (world = first camera)
        intr:  (frames, 3, 3)   matrix K for the h×w image (not for the original photo)
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
