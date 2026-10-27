# Indoor Planar Reconstruction

Detecting planes (walls, floor, ceiling) in a room from RGB images: the **VGGT** foundation model provides depth and cameras, then our aggregation pipeline extracts planes in every frame and merges them across views into a single set of room planes.

---

## 1. Problem

Existing approaches to planar reconstruction have two main problems:
- they either produce weak 3D geometry, or require a separate planar segmentation as a prerequisite;
- there is no common dataset and evaluation protocol, so different methods are hard to compare, especially on both planar segmentation quality and 3D geometry quality.

This project's goals are to implement a planar reconstruction pipeline from images and to define a dataset and an evaluation protocol with separate metrics for planar segmentation and for 3D geometry quality.

In this project we use *feed-forward* approach on top of VGGT. VGGT predicts depth and cameras, and our aggregation pipeline extracts planes from them.

### **Task**
Input: several RGB images of one room (different viewpoints). \
Output:
- for every frame, a mask of the pixels belonging to each plane;
- 3D parameters of every plane: normal `n` and offset `d` (plane equation `n·X + d = 0` in the camera coordinate system, units in mm);
- plane identifiers shared across all views of the room (the same wall seen in different frames gets the same ID).

## 2. Data

We use the **[Structured3D](https://structured3d-dataset.org/)** dataset: synthetic photo-realistic interiors with accurate room-layout annotations.

For each view we use:

| File | Purpose |
|---|---|
| `rgb.png` | method input |
| `depth.png` | GT depth (for evaluation and GT masks) |
| `semantic.png` | semantics (optional, for the GT protocol) |
| `layout.json` | plane polygons: the basis of the GT |
| `camera.txt` | camera parameters |

**Plane GT** is built from `layout.json`, but a pixel enters the mask only if it **geometrically lies on the plane** according to GT depth (tolerance `max(20 mm, 1% of depth)`). Furniture, mirrors, and views through windows are therefore excluded from the GT planes.

**Split** (`split.json`, split by scene so that rooms from one scene never end up in different parts):

| train | val | test |
|---|---|---|
| 120 scenes | 30 scenes | 50 scenes |

The first 10 scenes are deliberately assigned to train, since the method was developed on them. Parameters are tuned on `val`; `test` is run once at the very end.

> The dataset itself is not included in this repository: get it separately from the official Structured3D website.

## 3. Approach

```
RGB (N views of a room)
   │
   ▼
VGGT-1B ──► depth, confidence, cameras (extr/intr)
   │
   ▼
scale to mm ──► 3D points of the frame
   │
   ▼
Sequential RANSAC ─► cleaning ─► merging pieces ─► boundary filter     (within a frame)
   │
   ▼
planes to world coordinates ─► merging across views                    (within a room)
   │
   ▼
evaluation against GT
```

**VGGT** (`vggt_model.py`). The pretrained VGGT-1B model (`facebook/VGGT-1B`) takes all views of a room together and returns depth, a confidence map, and cameras. Images are resized to 518×294, so plane masks are computed at that size and scaled back to the photo size for evaluation.

**Scale** (`pipeline.py`). VGGT depth is given in an arbitrary scale. Two modes:
- `self` (default, honest): the median depth of reliable pixels in the room is set to 2500 mm; GT is not used. The GT scale is applied **only** to plane offsets during evaluation, so that the scale error does not corrupt the distance metric;

**Planes in a frame** (`ransac.py`, `aggregation.py`):
1. drop the 30% of pixels with the lowest confidence (`--conf-pct`);
2. **Sequential RANSAC**: find the best plane, remove its points, repeat (up to 12 planes), refining each plane with PCA;
3. **Cleaning** (`clean_planes`): split into connected components, morphological opening, discard small fragments;
4. **Merging pieces** (`merge_coplanar`): merge fragments with nearly identical normal (< 5°) and offset;
5. **Boundary filter** (`filter_boundary`): keep only planes with almost no frame points behind them (≤ 3%). This removes furniture and other planes inside the room, leaving walls, floor, and ceiling.

**Merging across views** (`aggregation.to_world`, `merge_items`). Planes are transformed into world coordinates using the VGGT camera poses. Two planes from different frames are considered the same if the angle between normals is < 5°, the offset difference is < 100 mm, and their points touch (distance < 200 mm). Groups are built with union-find.

## 4. Metrics

Evaluation protocol (`metrics.py`):
- predicted planes are matched to GT with the Hungarian algorithm on **IoU**, threshold 0.5;
- IoU is computed only over pixels that GT considers planes (mirrors, doors, etc. are not penalized);
- GT planes smaller than 0.5% of the frame are ignored;
- **TP / FP / FN → Precision, Recall, F1** (aggregated over all frames), **mIoU** over matched pairs;
- 3D errors for matched pairs only: **normal angle** (°) and **offset error** (mm);
- **cross-view merging quality**: how many GT walls visible in 2+ frames were merged correctly, and how many groups wrongly merged different walls.

## 5. Current results

> All parameters are tuned on `val`. Final `test` numbers are added once, after the parameters are frozen.

### Ablation on val (self scale)

| Configuration | TP | FP | FN | Precision | Recall | F1 | mIoU | Angle, ° | Offset, mm |
|---|---|---|---|---|---|---|---|---|---|
| RANSAC + cleaning | 1155 | 6359 | 1186 | 0.15 | 0.49 | 0.23 | 0.77 | 2.5 | 78.1 |
| + merging pieces | 1307 | 4381 | 1034 | 0.23 | 0.56 | 0.33 | 0.78 | 2.45 | 77.7 |
| + boundary filter (main method) | 1041 | 1603 | 1300 | 0.39 | 0.44 | 0.42 | 0.78 | 1.74 | 67.6 |
| main method, GT scale (upper bound) | 1077 | 1657 | 1264 | 0.39 | 0.46 | 0.42 | 0.78 | 1.87 | 68.8 |

### Cross-view merging (val)

| Walls seen in 2+ frames | Merged correctly | Wrong merges |
|---|---|---|
| 278 | 203 (73%) | 0 |
### Limitations

- The `self` scale is heuristic (a fixed "typical" depth of 2.5 m), so plane distances carry a systematic error if the room is much larger or smaller than typical.
- In a quick smoke test on 2 rooms (7 frames), precision exceeded recall (0.52 vs 0.35): the method misses some planes. That test only verified that the run works and is not a result for evaluation.
- RANSAC, cleaning, and boundary-filter thresholds are set by hand and tuned on `val`.
- Quality depends entirely on VGGT depth; depth errors on mirrors, glass, and distant regions propagate into the planes.

### Status and plans

Planes are currently found heuristically (RANSAC + filters). The next step is a trained **planar head** on top of VGGT features, intended to replace part of the heuristic filters. It is under development and is not part of the results above.

## 6. How to run

Tested in **Google Colab (GPU)**. An **NVIDIA GPU with CUDA** is required: VGGT is loaded onto `cuda`, so the code does not run on CPU or Mac. On first run the VGGT-1B weights (several GB) are downloaded from Hugging Face.

### Colab (quick start)

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -q "numpy>=2.1,<2.3" huggingface_hub safetensors einops
!pip install -q --no-deps git+https://github.com/facebookresearch/vggt.git
!rm -rf /content/planes && git clone -q https://github.com/vusu4242/indoor_planar_reconstruction.git /content/planes

!cd /content && python -m planes.main --root "/content/drive/MyDrive/AI project/dataset" --split val --limit 2
```

If Colab asks to restart the runtime after installing `numpy`, restart it and run the cell again.

### Locally

The repository **must be named `planes`** (the code uses `from planes import ...`), and you run it from the parent folder:

```bash
git clone https://github.com/vusu4242/indoor_planar_reconstruction.git planes
pip install -r planes/requirements.txt     # better to install torch separately for your CUDA version
python -m planes.main --root /path/to/Structured3D --split val
```

### Main `main` parameters

| Parameter | Description |
|---|---|
| `--root` | dataset folder (`scene_*/room*/view*/...`) |
| `--split {train,val,test}` or `--scenes ...` | which scenes to process |
| `--limit N` | only the first N rooms (for a quick check) |
| `--scale-mode {self,oracle}` | scale without GT (honest) or from GT (upper bound) |
| `--ablation` | compare RANSAC / + merging pieces / + boundary filter |
| `--conf-pct` | percentage of least-confident pixels to drop |
| `--no-merge`, `--no-boundary` | turn off merging pieces / the boundary filter |
| `--tag`, `--out` | run name / results folder |

All parameters: `python -m planes.main --help`.

### Run outputs

Saved to `results/<tag>/`:

| File | Contents |
|---|---|
| `summary.csv` | overall metrics |
| `frames.csv` | metrics for every frame |
| `merge.csv` | merging of planes across views, per room |
| `config.json` | run parameters and the list of rooms |

## 7. Repository structure

```
planes/
├── data.py          Structured3D loading, cameras, back-projection
├── gt.py            building GT planes from layout.json
├── vggt_model.py    running VGGT
├── ransac.py        PCA plane fit and Sequential RANSAC
├── aggregation.py   cleaning, merging pieces, boundary filter, cross-view merging
├── metrics.py       evaluation metrics
├── pipeline.py      full pipeline for a room
├── main.py          command-line entry point
├── planar_head.py   experimental (in development)
├── split.json       train / val / test scene split
└── __init__.py
```

## 8. AI usage

| Tool | What it was used for |
|---|---|
| Claude (Anthropic) | Generating code for the pipeline; editing text for the project reports|

Also: the project uses the pretrained **VGGT-1B** model (Meta AI) without fine-tuning; the rest of the pipeline was written by us.

## 9. Contribution statement

Dataset preparation - Olena Tumak\
Baseline pipelinе(aggregation) - Yuliana Vus\
Experiments, parameter tuning - Yuliana Vus, Olena Tumak\
Confidence threshold experiments - Maksym Prokopets\
Planar head - Maksym Prokopets(in progress)\

## 10. References and license

- **VGGT**: Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025. <https://github.com/facebookresearch/vggt>
- **Structured3D**: Zheng et al., *Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling*, ECCV 2020. <https://structured3d-dataset.org/>

