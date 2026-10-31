"""Ten-fold cross-validation of the selected model.

All images of the dataset (train, valid and test of the Roboflow export) are
pooled and split into K folds stratified by the number of annotated berries:
images are sorted by berry count, grouped in consecutive blocks of K, and the
images of each block are assigned to the K folds in a shuffled order. In round
k, fold k is the test set, folds k+1 and k+2 (circularly) are the validation
set and the remaining folds are the training set. The validation set is used
only for early stopping and for choosing best.pt; the reported metrics are
computed on the test fold.

The split is deterministic. Its SHA-256 fingerprint is printed so that runs
made in different sessions can be checked against each other.

    python cross_validation.py split --dataset dataset --fold 3
    python cross_validation.py train --fold 3
    python cross_validation.py evaluate --fold 3 --run runs/cv/fold3
    python cross_validation.py summary --results runs/cv
"""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from grapeseg import settings
from grapeseg.data import SPLITS
from grapeseg.report import METRICS, best_epoch, metrics_dict, save_json, table

K = 10
SPLIT_SEED = 2026
BERRY_CLASS = 1  # names: ['bunch of grapes', 'grape']
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def group_key(stem):
    """'00094_jpg.rf.01f6...' -> '00094', so that copies of a photo stay together."""
    base = stem.split(".rf.")[0]
    return re.sub(r"_(jpg|jpeg|png|bmp|webp)$", "", base, flags=re.I)


def collect_groups(root):
    groups = defaultdict(list)
    for split in SPLITS:
        folder = root / split / "images"
        if not folder.is_dir():
            continue
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = root / split / "labels" / (image.stem + ".txt")
            groups[group_key(image.stem)].append((image, label if label.exists() else None))
    return groups


def berry_count(items):
    total = 0
    for _, label in items:
        if label:
            total += sum(1 for line in label.read_text().splitlines()
                         if line.split() and int(line.split()[0]) == BERRY_CLASS)
    return total


def assign_folds(density, k=K, seed=SPLIT_SEED):
    rng = random.Random(seed)
    order = sorted(density, key=lambda g: (density[g], g))
    fold_of = {}
    for start in range(0, len(order), k):
        folds = list(range(k))
        rng.shuffle(folds)
        for group, fold in zip(order[start:start + k], folds):
            fold_of[group] = fold
    return fold_of


def fingerprint(fold_of):
    text = "\n".join(f"{g},{fold_of[g]}" for g in sorted(fold_of))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def role(fold, test_fold, k=K):
    if fold == test_fold:
        return "test"
    if fold in {(test_fold + 1) % k, (test_fold + 2) % k}:
        return "val"
    return "train"


def _link(src, dst):
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def split(args):
    root = Path(args.dataset)
    groups = collect_groups(root)
    density = {g: berry_count(items) for g, items in groups.items()}
    fold_of = assign_folds(density)
    digest = fingerprint(fold_of)
    print(f"images: {sum(len(v) for v in groups.values())}   groups: {len(groups)}   fingerprint: {digest}")
    if args.expected and digest != args.expected:
        raise SystemExit(f"fingerprint {digest} differs from {args.expected}: the dataset is not the same")

    out = Path(args.out) / f"fold{args.fold}"
    shutil.rmtree(out, ignore_errors=True)
    counts = defaultdict(lambda: [0, 0])
    for group, items in groups.items():
        part = role(fold_of[group], args.fold)
        for sub in ("images", "labels"):
            (out / part / sub).mkdir(parents=True, exist_ok=True)
        for image, label in items:
            _link(image, out / part / "images" / image.name)
            if label:
                _link(label, out / part / "labels" / label.name)
        counts[part][0] += len(items)
        counts[part][1] += density[group]

    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    data = dict(path=str(out.resolve()), train="train/images", val="val/images", test="test/images",
                nc=len(names), names=names)
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    (out / "split.json").write_text(json.dumps(
        dict(fold=args.fold, fingerprint=digest, images={p: c[0] for p, c in counts.items()},
             berries={p: c[1] for p, c in counts.items()}), indent=2))

    with open(Path(args.out) / "folds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "fold", "berries", "images"])
        for group in sorted(fold_of):
            writer.writerow([group, fold_of[group], density[group], ";".join(i.name for i, _ in groups[group])])

    for part in ("train", "val", "test"):
        print(f"{part:<6}{counts[part][0]:>5} images{counts[part][1]:>8} berries")


def evaluate(fold, run_dir, data, batch):
    """Metrics of best.pt on the test fold."""
    from ultralytics import YOLO

    model = YOLO(str(Path(run_dir) / "weights" / "best.pt"))
    results = model.val(data=data, split="test", imgsz=settings.IMAGE_SIZE, batch=batch,
                        overlap_mask=False, max_det=settings.MAX_DETECTIONS, plots=False,
                        project=str(run_dir), name="test", exist_ok=True)
    return metrics_dict(results, model.names)


def training_arguments(args, data):
    # The seed is the same in every fold, so that the variation between folds
    # comes from the data partition.
    return dict(
        data=str(data),
        batch=args.batch,
        seed=0,
        deterministic=True,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=f"fold{args.fold}",
        save_period=25,
        **settings.SCHEDULE_VARIABILITY,
        **settings.TRAINING,
        **settings.ONLINE_AUGMENTATION,
    )


def train(args):
    from ultralytics import YOLO

    fold_dir = Path(args.folds) / f"fold{args.fold}"
    data = fold_dir / "data.yaml"
    if not data.exists():
        raise SystemExit(f"{data} not found; run the split step first")
    model = YOLO(args.model)
    results = model.train(**training_arguments(args, data))
    run_dir = Path(model.trainer.save_dir)
    write_record(args.fold, run_dir, fold_dir, args.project, args.batch, validation=metrics_dict(results, model.names))


def write_record(fold, run_dir, fold_dir, project, batch, validation=None):
    test = evaluate(fold, run_dir, str(fold_dir / "data.yaml"), batch)
    info = json.loads((fold_dir / "split.json").read_text())
    epoch, epochs_run = best_epoch(run_dir)
    record = dict(fold=fold, fingerprint=info["fingerprint"], images=info["images"],
                  best_epoch=epoch, epochs_run=epochs_run, test=test)
    if validation:
        record["validation"] = validation
    save_json(record, Path(project) / f"fold{fold}.json")
    print(table(test))


def evaluate_command(args):
    # For runs that finished through --resume, where the test step was not executed.
    write_record(args.fold, Path(args.run), Path(args.folds) / f"fold{args.fold}", args.project, args.batch)


def summary(args):
    records = {r["fold"]: r for r in (json.loads(p.read_text(encoding="utf-8"))
                                      for p in sorted(Path(args.results).glob("fold*.json")))}
    missing = sorted(set(range(K)) - set(records))
    if missing:
        print(f"missing folds: {missing}")
    fingerprints = {r["fingerprint"] for r in records.values()}
    if len(fingerprints) != 1:
        raise SystemExit(f"folds come from different splits: {fingerprints}")
    print(f"folds: {sorted(records)}   fingerprint: {fingerprints.pop()}")
    print(f"{'class':<18}{'type':<6}{'metric':<10}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
    for name in next(iter(records.values()))["test"]:
        for kind in ("box", "mask"):
            for metric in METRICS + ("F1",):
                values = [records[f]["test"][name][kind][metric] for f in sorted(records)]
                sd = statistics.stdev(values) if len(values) > 1 else float("nan")
                print(f"{name:<18}{kind:<6}{metric:<10}{statistics.mean(values):>9.4f}"
                      f"{sd:>9.4f}{min(values):>9.4f}{max(values):>9.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("split", help="build the folders of one fold")
    p.add_argument("--dataset", required=True, help="prepared dataset with train, valid and test folders")
    p.add_argument("--fold", type=int, required=True, choices=range(K))
    p.add_argument("--out", default="folds")
    p.add_argument("--expected", help="fingerprint of a previous split, checked before building the fold")
    p.set_defaults(func=split)

    for name, func, text in (("train", train, "train one fold and evaluate it on the test fold"),
                             ("evaluate", evaluate_command, "evaluate an already trained fold on its test fold")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--fold", type=int, required=True, choices=range(K))
        p.add_argument("--folds", default="folds")
        p.add_argument("--project", default="runs/cv")
        p.add_argument("--batch", type=int, default=8)
        if name == "train":
            p.add_argument("--model", default="yolo26l-seg.pt")
            p.add_argument("--device", default="0")
            p.add_argument("--workers", type=int, default=11)
        else:
            p.add_argument("--run", required=True, help="run directory with weights/best.pt")
        p.set_defaults(func=func)

    p = sub.add_parser("summary", help="mean and standard deviation over the folds")
    p.add_argument("--results", default="runs/cv")
    p.set_defaults(func=summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
