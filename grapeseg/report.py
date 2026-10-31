"""Extraction of box and mask metrics from Ultralytics validation results."""

import csv
import json
from pathlib import Path

METRICS = ("P", "R", "mAP50", "mAP50_95")


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def metrics_dict(results, names):
    """Precision, recall, F1, mAP50 and mAP50-95 for boxes and masks.

    Per-class F1 is computed from the precision and recall reported by the
    library; F1 over both classes is the mean of the per-class values, the
    same criterion the library applies to precision and recall.
    """
    def block(m, i=None):
        if i is None:
            values = dict(P=m.mp, R=m.mr, mAP50=m.map50, mAP50_95=m.map)
        else:
            values = dict(P=m.p[i], R=m.r[i], mAP50=m.ap50[i], mAP50_95=m.ap[i])
        return {k: round(float(v), 5) for k, v in values.items()}

    out = {}
    for i, c in enumerate(results.box.ap_class_index):
        out[names[int(c)]] = {"box": block(results.box, i), "mask": block(results.seg, i)}
    out["all"] = {"box": block(results.box), "mask": block(results.seg)}
    for kind in ("box", "mask"):
        per_class = []
        for name, values in out.items():
            values[kind]["F1"] = round(f1(values[kind]["P"], values[kind]["R"]), 5)
            if name != "all":
                per_class.append(values[kind]["F1"])
        out["all"][kind]["F1"] = round(sum(per_class) / len(per_class), 5)
    return out


def table(metrics):
    """Plain-text table of a metrics_dict result."""
    header = f"{'class':<18}" + "".join(f"{k + '(' + t[0].upper() + ')':>12}" for t in ("box", "mask") for k in METRICS + ("F1",))
    rows = [header]
    for name in sorted(metrics, key=lambda n: (n != "all", n)):
        rows.append(f"{name:<18}" + "".join(f"{metrics[name][t][k]:>12.4f}" for t in ("box", "mask") for k in METRICS + ("F1",)))
    return "\n".join(rows)


def best_epoch(run_dir):
    """Epoch of best.pt, using the fitness Ultralytics applies to segmentation."""
    with open(Path(run_dir) / "results.csv", newline="") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]

    def fitness(row):
        return sum(0.1 * float(row[f"metrics/mAP50({t})"]) + 0.9 * float(row[f"metrics/mAP50-95({t})"])
                   for t in ("B", "M"))

    best = max(rows, key=fitness)
    return int(float(best["epoch"])), len(rows)


def save_json(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
