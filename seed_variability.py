"""Variability of the selected model with respect to the random seed.

The selected configuration (YOLOv26l, online augmentation, no offline
augmentation) is trained again with a different seed per run, keeping the
data partition fixed. Deterministic mode is disabled on purpose, so that the
non-deterministic GPU operations also contribute to the measured variation.

    python seed_variability.py train --data dataset/data.yaml --seed 3
    python seed_variability.py train --seed 3 --resume
    python seed_variability.py summary --results runs/seeds

Each run writes seed<k>.json; the summary step reports the mean and the
sample standard deviation of every metric over the available seeds.
"""

import argparse
import json
import statistics
from pathlib import Path

from grapeseg import settings
from grapeseg.report import METRICS, best_epoch, metrics_dict, save_json, table


def training_arguments(args):
    return dict(
        data=args.data,
        batch=args.batch,
        seed=args.seed,
        deterministic=False,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=f"seed{args.seed}",
        save_period=25,
        **settings.SCHEDULE_VARIABILITY,
        **settings.TRAINING,
        **settings.ONLINE_AUGMENTATION,
    )


def train(args):
    from ultralytics import YOLO

    if args.resume:
        model = YOLO(str(Path(args.project) / f"seed{args.seed}" / "weights" / "last.pt"))
        results = model.train(resume=True)
    else:
        if not args.data:
            raise SystemExit("--data is required for a new run")
        model = YOLO(args.model)
        results = model.train(**training_arguments(args))
    run_dir = Path(model.trainer.save_dir)
    metrics = metrics_dict(results, model.names)
    epoch, epochs_run = best_epoch(run_dir)
    record = dict(seed=args.seed, model=args.model, best_epoch=epoch, epochs_run=epochs_run, validation=metrics)
    save_json(record, Path(args.project) / f"seed{args.seed}.json")
    print(table(metrics))


def summary(args):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.results).glob("seed*.json"))]
    if len(records) < 2:
        raise SystemExit("at least two seed results are needed")
    print(f"seeds: {sorted(r['seed'] for r in records)}")
    print(f"{'class':<18}{'type':<6}{'metric':<10}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
    for name in records[0]["validation"]:
        for kind in ("box", "mask"):
            for metric in METRICS + ("F1",):
                values = [r["validation"][name][kind][metric] for r in records]
                print(f"{name:<18}{kind:<6}{metric:<10}{statistics.mean(values):>9.4f}"
                      f"{statistics.stdev(values):>9.4f}{min(values):>9.4f}{max(values):>9.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train the selected configuration with one seed")
    p.add_argument("--data", help="data.yaml of the dataset without offline augmentation")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--resume", action="store_true", help="resume the interrupted run of this seed from its last.pt")
    p.add_argument("--model", default="yolo26l-seg.pt")
    p.add_argument("--batch", type=int, default=8, help="reduced by Ultralytics if the GPU memory is insufficient")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--project", default="runs/seeds")
    p.set_defaults(func=train)

    p = sub.add_parser("summary", help="mean and standard deviation over the seeds")
    p.add_argument("--results", default="runs/seeds", help="folder with the seed<k>.json files")
    p.set_defaults(func=summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
