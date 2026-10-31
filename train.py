"""Train a YOLO instance segmentation model on the grape dataset.

Used in the comparison between architectures (YOLOv11 and YOLOv26), forms of
data augmentation (offline and online) and model sizes (s, m, l and x).
The metrics of best.pt on the validation set are written to metrics.json in
the run directory.
"""

import argparse
from pathlib import Path

from grapeseg import settings
from grapeseg.report import best_epoch, metrics_dict, save_json, table


def training_arguments(args):
    schedule = settings.SCHEDULE_OFFLINE if args.offline else settings.SCHEDULE_NO_OFFLINE
    augmentation = settings.NO_ONLINE_AUGMENTATION if args.no_online else settings.ONLINE_AUGMENTATION
    return {
        **settings.TRAINING,
        **augmentation,
        "data": args.data,
        "epochs": args.epochs or schedule["epochs"],
        "patience": args.patience or schedule["patience"],
        "batch": args.batch,
        "seed": args.seed,
        "deterministic": True,
        "device": args.device,
        "workers": args.workers,
        "cache": False if args.cache == "none" else args.cache,
        "project": args.project,
        "name": args.name,
        "save_period": args.save_period,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="data.yaml of the prepared dataset")
    parser.add_argument("--model", default="yolo26l-seg.pt", help="pretrained weights, e.g. yolo11s-seg.pt or yolo26x-seg.pt")
    parser.add_argument("--offline", action="store_true", help="the dataset contains offline augmentation (150 epochs, patience 40)")
    parser.add_argument("--no-online", action="store_true", help="disable online augmentation")
    parser.add_argument("--epochs", type=int, help="override the epoch limit")
    parser.add_argument("--patience", type=int, help="override the early-stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="reduced by Ultralytics if the GPU memory is insufficient")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=11, help="data loader processes; changes the order of the random augmentations")
    parser.add_argument("--cache", choices=("disk", "none"), default="disk", help="cache of decoded images")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", help="run name, required unless --resume is given")
    parser.add_argument("--save-period", type=int, default=10, help="checkpoint interval in epochs")
    parser.add_argument("--resume", metavar="LAST_PT", help="resume an interrupted run from its last.pt")
    args = parser.parse_args()
    if not args.resume and not args.name:
        parser.error("--name is required for a new run")

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(**training_arguments(args))

    run_dir = Path(model.trainer.save_dir)
    metrics = metrics_dict(results, model.names)
    epoch, epochs_run = best_epoch(run_dir)
    save_json(
        dict(model=args.model, offline=args.offline, online=not args.no_online,
             best_epoch=epoch, epochs_run=epochs_run, validation=metrics),
        run_dir / "metrics.json",
    )
    print(table(metrics))


if __name__ == "__main__":
    main()
