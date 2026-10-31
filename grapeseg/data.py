"""Dataset download and label preparation."""

import os
from collections import Counter
from pathlib import Path

SPLITS = ("train", "valid", "test")


def download_dataset(workspace, project, version, location):
    """Download a Roboflow dataset version in YOLO segmentation format.

    The API key is read from the ROBOFLOW_API_KEY environment variable.
    Returns the path of the generated data.yaml.
    """
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("set the ROBOFLOW_API_KEY environment variable")
    rf = Roboflow(api_key=api_key)
    rf.workspace(workspace).project(project).version(version).download("yolov11", location=str(location))
    return Path(location) / "data.yaml"


def _parse(line):
    parts = line.split()
    if not parts:
        return None, None, None
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        return parts[0], values, "box"
    if len(values) >= 6 and len(values) % 2 == 0:
        return parts[0], values, "polygon"
    return parts[0], values, None


def _box_to_polygon(values):
    xc, yc, w, h = values
    x1, x2 = xc - w / 2, xc + w / 2
    y1, y2 = yc - h / 2, yc + h / 2
    return [min(max(p, 0.0), 1.0) for p in (x1, y1, x2, y1, x2, y2, x1, y2)]


def convert_boxes_to_polygons(root):
    """Rewrite bounding-box label lines as four-vertex polygons.

    The Roboflow export stores part of the instances as boxes. A label file
    that mixes boxes and polygons is discarded by Ultralytics in segmentation
    training, so every box is converted to the polygon of its corners.
    Returns the number of converted lines per class.
    """
    root = Path(root)
    converted = Counter()
    for split in SPLITS:
        folder = root / split / "labels"
        if not folder.is_dir():
            continue
        for path in folder.glob("*.txt"):
            lines, changed = [], False
            for line in path.read_text().splitlines():
                cls, values, kind = _parse(line)
                if kind == "box":
                    points = _box_to_polygon(values)
                    lines.append(cls + " " + " ".join(f"{p:.6f}" for p in points))
                    converted[cls] += 1
                    changed = True
                elif line.strip():
                    lines.append(line.strip())
            if changed:
                path.write_text("\n".join(lines) + "\n")
    # Stale label caches would keep the discarded files out of training.
    for cache in root.rglob("*.cache"):
        cache.unlink()
    return converted


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download a dataset version and prepare its labels.")
    parser.add_argument("--workspace", default="hugo-leal-cn7iw")
    parser.add_argument("--project", default="grapes-mic3h")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--out", default="dataset")
    args = parser.parse_args()

    data = download_dataset(args.workspace, args.project, args.version, args.out)
    converted = convert_boxes_to_polygons(args.out)
    print(f"data.yaml: {data}")
    print(f"box labels converted to polygons: {dict(converted) or 'none'}")


if __name__ == "__main__":
    main()
