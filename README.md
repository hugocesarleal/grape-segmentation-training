# Grape Segmentation Training

Training and evaluation protocol for YOLO instance segmentation models that
detect grape bunches and individual berries in field images. The code
reproduces the three experimental stages of the study:

| Script | Stage |
|---|---|
| `train.py` | Comparison of architectures (YOLOv11 and YOLOv26), offline and online data augmentation, and model sizes |
| `seed_variability.py` | Repeated training of the selected model with different random seeds |
| `cross_validation.py` | Ten-fold cross-validation of the selected model, stratified by the number of berries per image |

The shared configuration (hyperparameters, augmentation and training
schedules) is in `grapeseg/settings.py`, so that every stage trains under the
same conditions.

## Requirements

Python 3.10 or later and a CUDA GPU. The experiments ran on Google Colab with
NVIDIA L4 (24 GB) and A100 (80 GB) GPUs.

```bash
pip install -r requirements.txt
```

The Ultralytics version is fixed at 8.4.157. Other versions may change default
values or the way precision, recall and F1 are reported.

## Dataset

The dataset contains 530 field images of green and purple grapes from the
experimental vineyard of the Polytechnic Institute of Bragança, annotated with
two classes, `bunch of grapes` and `grape`. It is private and hosted on
Roboflow. Version 14 of the Roboflow project is the export without offline
augmentation, split into 371 training, 106 validation and 53 test images.

```bash
export ROBOFLOW_API_KEY=...
python -m grapeseg.data --version 14 --out dataset
```

The preparation step converts the instances exported as bounding boxes into
four-vertex polygons. Without it, Ultralytics discards label files that mix
boxes and polygons in segmentation training.

## Usage

### Stage 1 and 2: model comparison

```bash
# YOLOv26s with online augmentation, dataset without offline augmentation
python train.py --data dataset/data.yaml --model yolo26s-seg.pt --name yolo26s-online

# same model on the dataset with tenfold offline augmentation
python train.py --data dataset-offline/data.yaml --model yolo26s-seg.pt --offline --name yolo26s-offline-online

# no augmentation at all
python train.py --data dataset/data.yaml --model yolo26s-seg.pt --no-online --name yolo26s-none
```

`--offline` selects the schedule of 150 epochs with patience 40; otherwise the
limit is 600 epochs with patience 80. Each run writes `metrics.json` with the
validation metrics of `best.pt`. An interrupted run is resumed with
`--resume runs/<name>/weights/last.pt`.

### Stage 3: variability with respect to the seed

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python seed_variability.py train --data dataset/data.yaml --seed $seed
done
python seed_variability.py summary --results runs/seeds
```

The selected configuration is YOLOv26l with online augmentation and without
offline augmentation. Deterministic mode is disabled, so the measured
variation also includes the non-deterministic GPU operations. The limit is
600 epochs with patience 120.

### Stage 3: cross-validation

```bash
for fold in 0 1 2 3 4 5 6 7 8 9; do
    python cross_validation.py split --dataset dataset --fold $fold
    python cross_validation.py train --fold $fold
done
python cross_validation.py summary --results runs/cv
```

In round k, fold k is the test set and folds k+1 and k+2 (circularly) form the
validation set, which is used only for early stopping and for choosing
`best.pt`. The metrics reported for each round are computed on the test fold.
The split is deterministic and prints a SHA-256 fingerprint; passing it as
`--expected` in later sessions stops the script if the dataset differs. The
split used in the study, built from version 14 of the dataset, has the
fingerprint `c6165896dbcf`.

An interrupted seed or fold is resumed with the same command plus `--resume`;
the record of the run, including the test evaluation of a fold, is written when
the training ends.

## Google Colab

The `notebooks` folder has one notebook per stage. Each notebook clones this
repository, prepares the dataset and calls the scripts above, so the runs are
the same as from the command line. Outputs are written to Google Drive, and the
Roboflow API key is read from the Colab secret `ROBOFLOW_API_KEY`.

| Stage | Notebook |
|---|---|
| Model comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hugocesarleal/grape-segmentation-training/blob/main/notebooks/model_comparison.ipynb) |
| Seed variability | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hugocesarleal/grape-segmentation-training/blob/main/notebooks/seed_variability.ipynb) |
| Cross-validation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hugocesarleal/grape-segmentation-training/blob/main/notebooks/cross_validation.ipynb) |

## Reproducing the runs of the study

The settings below were taken from the arguments recorded by Ultralytics in
each run. For every run with a complete record, the arguments generated by
these scripts match the recorded ones in all parameters. `Batch` is the value
requested; Ultralytics halves it when the GPU runs out of memory, so the
effective value also depends on the GPU.

| Model | Offline | Online | Batch | Options | GPU | Ultralytics |
|---|---|---|---|---|---|---|
| YOLOv11s | yes | yes | 8 | `--workers 8 --cache none` | L4 | 8.4.153 |
| YOLOv11s | no | yes | not recorded | | L4 | 8.4.155 |
| YOLOv11s | yes | no | not recorded | | L4 | 8.4.155 |
| YOLOv11s | no | no | 8 | | L4 | 8.4.157 |
| YOLOv26s | no | yes | not recorded | | L4 | 8.4.155 |
| YOLOv26s | no | no | not recorded | | L4 | 8.4.157 |
| YOLOv26s | yes | no | not recorded | | L4 | 8.4.155 |
| YOLOv26s | yes | yes | 8 | | L4 | not recorded |
| YOLOv26m | yes | yes | 12 | | L4 | 8.4.157 |
| YOLOv26m | no | yes | 30 | | A100 | 8.4.157 |
| YOLOv26l | yes | yes | 30 | | A100 | 8.4.157 |
| YOLOv26l | no | yes | 30 | | A100 | 8.4.157 |
| YOLOv26x | yes | yes | 32 | | A100 | 8.4.157 |
| YOLOv26x | no | yes | 12 | | L4 | 8.4.157 |
| YOLOv26l, seeds 0 to 9 | no | yes | 8 | | L4 | 8.4.157 |
| YOLOv26l, folds 0 to 9 | no | yes | 8 | | L4 | 8.4.157 |

Runs with deterministic mode give the same result only with the same GPU,
CUDA and library versions. The seed runs disable deterministic mode by design
and are not expected to repeat exactly.

## Metrics

Precision, recall, mAP@50 and mAP@50-95 are reported by Ultralytics for boxes
and masks. F1 is computed from precision and recall, per class, and averaged
over the two classes. Validation and test use `overlap_mask=False`, which keeps
the mask of a bunch intact where berries overlap it.

## License

The code is released under the MIT License. Ultralytics, used for training and
evaluation, is distributed under its own license (AGPL-3.0).

## Related repositories

- Grape Frame Extractor: extraction of frames from the vineyard videos.
- Grape Dataset Deduplication: removal of redundant images by perceptual hashing.
