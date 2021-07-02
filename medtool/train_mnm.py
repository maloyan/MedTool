import json
import sys

import albumentations as A
import nibabel as nib
import numpy as np
import segmentation_models_pytorch as smp
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import medtool.data.feta_dataset as feta_dataset
import medtool.data.mnm_dataset as mnm_dataset

with open(sys.argv[1], "r") as f:
    config = json.load(f)

wandb.init(config=config, project=config["project"])

transform = A.Compose(
    [
        A.Resize(height=256, width=256, always_apply=True, p=1),
        A.OneOf(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.RandomGamma(p=1),
                A.RandomBrightnessContrast(p=1),
            ],
            p=0.2,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=180, p=0.2
        ),
        A.CoarseDropout(p=0.1),
    ],
    p=1,
)


(
    train_data_path,
    train_labels_path,
    valid_data_path,
    valid_labels_path,
) = mnm_dataset.get_train_val_path(train_dir=config["train_dir"], split=0.9)

train_data = []
train_labels = []
val_data = []
val_labels = []

for i in train_data_path:
    train_data.extend(np.moveaxis(np.asarray(nib.load(i).dataobj), -1, 0))

for i in train_labels_path:
    train_labels.extend(np.moveaxis(np.asarray(nib.load(i).dataobj), -1, 0))

for i in valid_data_path:
    val_data.extend(np.moveaxis(np.asarray(nib.load(i).dataobj), -1, 0))

for i in valid_labels_path:
    val_labels.extend(np.moveaxis(np.asarray(nib.load(i).dataobj), -1, 0))


train_dataset = mnm_dataset.MnM(
    train_data,
    train_labels,
    augmentation=transform,
)
valid_dataset = mnm_dataset.MnM(val_data, val_labels, augmentation=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    drop_last=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    drop_last=False,
)
if config["model"] == "unetpp":
    model = smp.UnetPlusPlus(
        encoder_name=config["backbone"],
        encoder_weights=config["pretrained"],
        in_channels=config["in_channels"],
        classes=config["classes"],
        activation=config["activation"],
    )
elif config["model"] == "unet":
    model = smp.Unet(
        encoder_name=config["backbone"],
        encoder_weights=config["pretrained"],
        in_channels=config["in_channels"],
        classes=config["classes"],
        activation=config["activation"],
    )

model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
print("PARALLEL")
loss = smp.utils.base.SumOfLosses(
    smp.utils.losses.DiceLoss(), smp.utils.losses.BCELoss()
)

metrics = [smp.utils.metrics.IoU(threshold=0.5)]

optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=config["lr"]),
    ]
)

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=config["device"],
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=config["device"],
    verbose=True,
)

min_loss = 100
patience = 0
print("TRAINING")
for _ in range(150):
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    wandb.log(
        {
            "train_loss": train_logs["dice_loss + bce_loss"],
            "val_loss": valid_logs["dice_loss + bce_loss"],
            "IoU_train": train_logs["iou_score"],
            "IoU_val": valid_logs["iou_score"],
        }
    )

    optimizer.param_groups[0]["lr"] *= config["decay"]
    if min_loss > valid_logs["dice_loss + bce_loss"]:
        min_loss = valid_logs["dice_loss + bce_loss"]
        torch.save(model.module, f"models/{config['model_name']}.pt")
        print("Model saved!")
        patience = 0
    else:
        patience += 1

    if patience == config["patience"]:
        break
