"""
Точка входу: запуск пайплайну (варіант 1) з командного рядка.

Приклади (у Colab, з папки, ВСЕРЕДИНІ якої лежить пакет planes, тобто з /content):

    # базовий запуск на val-сценах з split.json
    !cd /content && python -m planes.main --root /content/data --split val

    # конкретні сцени, перші 3 кімнати, масштаб за GT (верхня межа)
    !cd /content && python -m planes.main --root /content/data --scenes scene_00000 --limit 3 --scale-mode oracle

    # абляції: RANSAC -> + злиття шматків -> + фільтр меж
    !cd /content && python -m planes.main --root /content/data --split val --ablation

Результати пишуться в --out (за замовчуванням results/<назва запуску>/):
    summary.csv  — підсумок (один рядок на конфігурацію)
    frames.csv   — метрики по кожному кадру
    merge.csv    — злиття площин між кадрами по кімнатах
    config.json  — з якими параметрами запускали
"""
import argparse
import json
import os
import time

import pandas as pd

from planes import data, pipeline, vggt_model

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description="Площинні примітиви: варіант 1 (VGGT + агрегація)")
    ap.add_argument("--root", required=True, help="папка з даними (scene_*/room*/view*/...)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--split", choices=["dev", "val", "test"], help="набір сцен з split.json")
    g.add_argument("--scenes", nargs="+", help="конкретні сцени, напр. scene_00000 scene_00001")
    ap.add_argument("--split-file", default=os.path.join(HERE, "split.json"))
    ap.add_argument("--limit", type=int, default=None, help="взяти лише перші N кімнат")
    ap.add_argument("--scale-mode", choices=["self", "oracle"], default="self")
    ap.add_argument("--conf-pct", type=float, default=30)
    ap.add_argument("--no-merge", action="store_true", help="вимкнути злиття шматків у кадрі")
    ap.add_argument("--no-boundary", action="store_true", help="вимкнути фільтр меж")
    ap.add_argument("--max-behind", type=float, default=0.03)
    ap.add_argument("--behind-max-mm", type=float, default=None)
    ap.add_argument("--footprint-margin-mm", type=float, default=None,
                    help="фільтр меж лише в межах площини (+ запас, мм); без параметра — як раніше")
    ap.add_argument("--order", choices=["ransac_first", "segment_first"], default="ransac_first",
                    help="спершу RANSAC по кадру чи спершу різати на області")
    ap.add_argument("--seg-angle-deg", type=float, default=10, help="segment_first: поріг кута нормалей")
    ap.add_argument("--seg-depth-rel", type=float, default=0.03, help="segment_first: поріг стрибка глибини")
    ap.add_argument("--ransac-min-frac", type=float, default=0.01,
                    help="RANSAC: мінімальна частка точок у площині (зупинка пошуку)")
    ap.add_argument("--clean-min-frac", type=float, default=0.005,
                    help="очищення: мінімальний розмір шматка як частка кадру")
    ap.add_argument("--open-k", type=int, default=3, help="ядро opening в очищенні (0 = вимкнено)")
    ap.add_argument("--close-k", type=int, default=0, help="ядро closing в очищенні (0 = вимкнено)")
    ap.add_argument("--gt-semantic", action="store_true", help="GT з семантикою (без вікон, жалюзі)")
    ap.add_argument("--ablation", action="store_true",
                    help="порівняти RANSAC / + злиття шматків / + фільтр меж")
    ap.add_argument("--out", default=None, help="куди писати результати")
    ap.add_argument("--tag", default=None, help="назва запуску (для папки результатів)")
    return ap.parse_args()


def pick_rooms(args):
    data.ROOT = args.root
    data._FRAMES = None
    if args.split:
        with open(args.split_file) as f:
            split = json.load(f)
        on_disk = set(data.list_scenes())
        scenes = [s for s in split[args.split] if s in on_disk]
        missing = len(split[args.split]) - len(scenes)
        if missing:
            print(f"увага: {missing} сцен з '{args.split}' немає в {args.root}, пропускаю їх")
    elif args.scenes:
        scenes = args.scenes
    else:
        scenes = data.list_scenes()
    rooms = data.list_rooms(scenes)
    return (rooms[: args.limit] if args.limit else rooms), scenes


def main():
    args = parse_args()
    rooms, scenes = pick_rooms(args)
    if not rooms:
        raise SystemExit("кімнат не знайдено — перевірте --root і назви сцен")
    print(f"сцен: {len(scenes)}, кімнат: {len(rooms)}")

    base = dict(conf_pct=args.conf_pct, scale_mode=args.scale_mode, gt_semantic=args.gt_semantic,
                max_behind=args.max_behind, behind_max_mm=args.behind_max_mm,
                open_k=args.open_k, close_k=args.close_k,
                footprint_margin_mm=args.footprint_margin_mm,
                ransac_min_frac=args.ransac_min_frac, clean_min_frac=args.clean_min_frac,
                order=args.order, seg_angle_deg=args.seg_angle_deg, seg_depth_rel=args.seg_depth_rel)
    if args.ablation:
        configs = {"RANSAC": dict(inframe_merge=False, boundary=False),
                   "+ злиття шматків": dict(inframe_merge=True, boundary=False),
                   "+ злиття шматків + фільтр меж": dict(inframe_merge=True, boundary=True)}
    else:
        configs = {"основний": dict(inframe_merge=not args.no_merge, boundary=not args.no_boundary)}

    tag = args.tag or f'{args.split or "custom"}_{args.scale_mode}_{time.strftime("%m%d-%H%M")}'
    out = args.out or os.path.join("results", tag)
    os.makedirs(out, exist_ok=True)

    model = vggt_model.load_model()
    summary, all_frames, all_merge = {}, [], []
    for name, kw in configs.items():
        t0 = time.time()
        summ, frames_df, merge_df = pipeline.run_all(model, rooms=rooms, **base, **kw)
        summ["кадрів"] = len(frames_df)
        summ["час_с"] = round(time.time() - t0)
        summary[name] = summ
        all_frames.append(frames_df.assign(конфігурація=name))
        all_merge.append(merge_df.assign(конфігурація=name))
        print(f"готово: {name}  F1={summ['F1']}")

    table = pd.DataFrame(summary).T
    print()
    print(table.to_string())
    table.to_csv(os.path.join(out, "summary.csv"))
    pd.concat(all_frames).to_csv(os.path.join(out, "frames.csv"), index=False)
    pd.concat(all_merge).to_csv(os.path.join(out, "merge.csv"), index=False)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "rooms": rooms}, f, indent=1, ensure_ascii=False)
    print(f"\nрезультати збережено в {out}")


if __name__ == "__main__":
    main()
