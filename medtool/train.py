import json
import sys

import nibabel as nib
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import medtool.data.feta_dataset as feta_dataset

with open(sys.argv[1], 'r') as f:
    config = json.load(f)

wandb.init(config=config, project='feta')

transform = A.Compose([
    A.CropNonEmptyMaskIfExists(height=128, width=128, p=0.2),
    A.OneOf([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], p=0.5),
    A.MedianBlur(blur_limit=3, p=0.01),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.05),
    A.CoarseDropout(p=0.05),
    A.Resize(height=256, width=256, always_apply=True, p=1)
], p=0.8)


data, labels = feta_dataset.get_path(config['data_path'])

train_data_path, valid_data_path     = data[:65], data[65:]
train_labels_path, valid_labels_path = labels[:65], labels[65:]

train_data   = np.array([np.moveaxis(np.asarray(nib.load(i).dataobj), -1, 0) for i in train_data_path]).reshape(-1, 256, 256)
train_labels = np.array([np.moveaxis(np.asarray(nib.load(i).dataobj), -1, 0) for i in train_labels_path]).reshape(-1, 256, 256)

train_data_new = []
train_labels_new = []

for i, j in tqdm(zip(train_data, train_labels), total=len(train_data_path)):
    if j.sum() == 0:
        rnd = np.random.rand()
        if rnd > 0.7:
            train_data_new.append(i)
            train_labels_new.append(j)
    else:
        train_data_new.append(i)
        train_labels_new.append(j)

train_dataset = feta_dataset.FeTA(
    np.array(train_data_new).reshape(-1, 256, 256),
    np.array(train_labels_new).reshape(-1, 256, 256),
    augmentation = transform
)
valid_dataset = feta_dataset.FeTA(
    np.array([np.moveaxis(np.asarray(nib.load(i).dataobj), -1, 0) for i in valid_data_path]).reshape(-1, 256, 256), 
    np.array([np.moveaxis(np.asarray(nib.load(i).dataobj), -1, 0) for i in valid_labels_path]).reshape(-1, 256, 256), 
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=config['num_workers'], 
    drop_last=True
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=config['num_workers'], 
    drop_last=False
)

model = smp.UnetPlusPlus(
    encoder_name    = config["model"], 
    encoder_weights = config["pretrained"],
    in_channels     = config["in_channels"],
    classes         = config["classes"],
    activation      = config["activation"]
)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
print("PARALLEL")
loss = smp.utils.base.SumOfLosses(
    smp.utils.losses.DiceLoss(),
    smp.utils.losses.BCELoss()
)

metrics = [
    smp.utils.metrics.IoU(threshold=0.5)
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=config['device'],
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=config['device'],
    verbose=True,
)

max_score = 0
patience = 0
print("TRAINING")
for _ in range(150):
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    wandb.log({
        "train_loss" : train_logs['dice_loss + bce_loss'],
        "val_loss"   : valid_logs['dice_loss + bce_loss'],
        "IoU_train"  : train_logs['iou_score'], 
        "IoU_val"    : valid_logs['iou_score']
    })

    optimizer.param_groups[0]['lr'] *= config['decay']
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model.module, f"models/{config['model_name']}.pt")
        print('Model saved!')
        patience = 0
    else:
        patience += 1
    
    if patience == config['patience']:
        break
