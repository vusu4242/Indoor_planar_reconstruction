"""
Runs the whole pipeline (VGGT+aggregation) with one command and saves the results.

Usage in Colab:
    !cd /content && python -m planes.main --root "/content/drive/MyDrive/AI project/dataset" --split val

Results (in results/<run name>/):
    summary.csv   overall metrics
    frames.csv    metrics for every frame
    merge.csv     merging of planes across frames, per room
    config.json   parameters of the run
"""
import argparse
import json
import os
import time

import pandas as pd

from planes import data, pipeline, vggt_model

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Command-line parameters (defaults are the same as in pipeline.process_room)."""
    ap = argparse.ArgumentParser(description="Planar primitives: Variant 1 (VGGT + aggregation)")
    ap.add_argument("--root", required=True, help="dataset folder (scene_*/room*/view*/...)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--split", choices=["train", "val", "test"], help="set of scenes from split.json")
    g.add_argument("--scenes", nargs="+", help="specific scenes, e.g. scene_00000 scene_00001")
    ap.add_argument("--split-file", default=os.path.join(HERE, "split.json"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rooms")
    ap.add_argument("--scale-mode", choices=["self", "oracle"], default="self",
                    help="self = scale without GT (honest); oracle = scale from GT (upper bound)")
    ap.add_argument("--conf-pct", type=float, default=30,
                    help="drop this percentage of least confident pixels in every frame")
    ap.add_argument("--no-merge", action="store_true", help="turn off merging of pieces within a frame")
    ap.add_argument("--no-boundary", action="store_true", help="turn off the boundary filter")
    ap.add_argument("--max-behind", type=float, default=0.03,
                    help="boundary filter: max fraction of points behind a plane")
    ap.add_argument("--behind-max-mm", type=float, default=None,
                    help="boundary filter: count only points at most this far behind a plane (mm)")
    ap.add_argument("--footprint-margin-mm", type=float, default=None,
                    help="boundary filter: count only points behind the plane itself (+ margin, mm)")
    ap.add_argument("--bilateral-d", type=int, default=0,
                    help="bilateral depth filter: neighbourhood size in pixels (0 = off)")
    ap.add_argument("--bilateral-sigma-rel", type=float, default=0.02,
                    help="bilateral: edge threshold as a fraction of depth")
    ap.add_argument("--ransac-min-frac", type=float, default=0.01,
                    help="RANSAC: minimal fraction of points in a plane (stops the search)")
    ap.add_argument("--clean-min-frac", type=float, default=0.005,
                    help="cleaning: minimal piece size as a fraction of the frame")
    ap.add_argument("--open-k", type=int, default=3, help="opening kernel in cleaning (0 = off)")
    ap.add_argument("--close-k", type=int, default=0, help="closing kernel at the end (0 = off)")
    ap.add_argument("--gt-semantic", action="store_true", help="GT with semantics (no windows, blinds)")
    ap.add_argument("--ablation", action="store_true",
                    help="compare RANSAC / + merging pieces / + boundary filter")
    ap.add_argument("--out", default=None, help="where to write the results")
    ap.add_argument("--tag", default=None, help="name of the run (for the results folder)")
    return ap.parse_args()


def pick_rooms(args):
    """Rooms to process: from --split, from --scenes, or all scenes in --root."""
    data.ROOT = args.root
    data._FRAMES = None
    if args.split:
        with open(args.split_file) as f:
            split = json.load(f)
        on_disk = set(data.list_scenes())
        scenes = [s for s in split[args.split] if s in on_disk]
        missing = len(split[args.split]) - len(scenes)
        if missing:
            print(f"warning: {missing} scenes from '{args.split}' are not in {args.root}, skipping them")
    elif args.scenes:
        scenes = args.scenes
    else:
        scenes = data.list_scenes()
    rooms = data.list_rooms(scenes)
    return (rooms[: args.limit] if args.limit else rooms), scenes


def main():
    """Runs the pipeline for each configuration and saves the result tables."""
    args = parse_args()
    rooms, scenes = pick_rooms(args)
    if not rooms:
        raise SystemExit("no rooms found: check --root and scene names")
    print(f"scenes: {len(scenes)}, rooms: {len(rooms)}")

    # parameters shared by all configurations
    base = dict(conf_pct=args.conf_pct, scale_mode=args.scale_mode, gt_semantic=args.gt_semantic,
                max_behind=args.max_behind, behind_max_mm=args.behind_max_mm,
                open_k=args.open_k, close_k=args.close_k,
                footprint_margin_mm=args.footprint_margin_mm,
                ransac_min_frac=args.ransac_min_frac, clean_min_frac=args.clean_min_frac,
                bilateral_d=args.bilateral_d, bilateral_sigma_rel=args.bilateral_sigma_rel)
    if args.ablation:
        configs = {"RANSAC": dict(inframe_merge=False, boundary=False),
                   "+ merging pieces": dict(inframe_merge=True, boundary=False),
                   "+ merging pieces + boundary filter": dict(inframe_merge=True, boundary=True)}
    else:
        configs = {"main": dict(inframe_merge=not args.no_merge, boundary=not args.no_boundary)}

    tag = args.tag or f'{args.split or "custom"}_{args.scale_mode}_{time.strftime("%m%d-%H%M")}'
    out = args.out or os.path.join("results", tag)
    os.makedirs(out, exist_ok=True)

    model = vggt_model.load_model()
    summary, all_frames, all_merge = {}, [], []
    for name, kw in configs.items():
        t0 = time.time()
        summ, frames_df, merge_df = pipeline.run_all(model, rooms=rooms, **base, **kw)
        summ["frames"] = len(frames_df)
        summ["time_s"] = round(time.time() - t0)
        summary[name] = summ
        all_frames.append(frames_df.assign(config=name))
        all_merge.append(merge_df.assign(config=name))
        print(f"done: {name}  F1={summ['F1']}")

    table = pd.DataFrame(summary).T
    print()
    print(table.to_string())
    table.to_csv(os.path.join(out, "summary.csv"))
    pd.concat(all_frames).to_csv(os.path.join(out, "frames.csv"), index=False)
    pd.concat(all_merge).to_csv(os.path.join(out, "merge.csv"), index=False)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "rooms": rooms}, f, indent=1, ensure_ascii=False)
    print(f"\nresults saved to {out}")


if __name__ == "__main__":
    main()
