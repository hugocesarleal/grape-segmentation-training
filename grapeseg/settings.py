"""Training configuration shared by the three experiments."""

IMAGE_SIZE = 1024
MAX_DETECTIONS = 300

# Hyperparameters kept identical across every run. Most are Ultralytics
# defaults, written out so that a different library version cannot change
# them silently.
TRAINING = dict(
    imgsz=IMAGE_SIZE,
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # Masks of overlapping instances are kept separate; with the default
    # (True) a berry mask overwrites the mask of the bunch that contains it.
    overlap_mask=False,
    mask_ratio=4,
    close_mosaic=10,
    amp=True,
    max_det=MAX_DETECTIONS,
    cache="disk",
    plots=True,
    val=True,
)

ONLINE_AUGMENTATION = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    erasing=0.4,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
)

NO_ONLINE_AUGMENTATION = {name: 0.0 for name in ONLINE_AUGMENTATION}
NO_ONLINE_AUGMENTATION["close_mosaic"] = 0

# Epoch limit and early-stopping patience. Runs on the dataset with offline
# augmentation see ten copies of each image per epoch, so they use fewer epochs.
SCHEDULE_OFFLINE = dict(epochs=150, patience=40)
SCHEDULE_NO_OFFLINE = dict(epochs=600, patience=80)
SCHEDULE_VARIABILITY = dict(epochs=600, patience=120)
