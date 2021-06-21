import os

import nibabel as nib
import torch
from torch.utils.data import Dataset


def get_data_path(dir):
    path_dict = {"image": [], "gt": []}
    for i in range(1, 161):
        patient_path = os.path.join(dir, f"{i:03}")
        for ls in ["LA", "SA"]:
            for key in ["ED", "ES"]:
                path_dict["image"].append(
                    os.path.join(patient_path, f"{i:003}_{ls}_{key}.nii.gz")
                )
                path_dict["gt"].append(
                    os.path.join(patient_path, f"{i:003}_{ls}_{key}_gt.nii.gz")
                )
    return path_dict["image"], path_dict["gt"]


def get_train_val_path(train_dir, val_dir=None, split=0.8):
    if val_dir is None:
        image, gt = get_data_path(train_dir)
        total = int(len(image) * split)
        return image[:total], gt[:total], image[total:], gt[total:]
    return get_data_path(train_dir), get_data_path(val_dir)


class MnM(Dataset):
    def __init__(self, data, masks=None, augmentation=None):
        super().__init__()
        self.data = data
        self.masks = masks
        self.augmentation = augmentation
        self.class_values = range(1, 4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]

        if self.masks is not None:
            mask = self.masks[item]
            masks = [(mask == v).astype("uint8") for v in self.class_values]
            if self.augmentation:
                sample = self.augmentation(image=image, masks=masks)
                image, masks = sample["image"], sample["masks"]

            return torch.tensor([image], dtype=torch.float), torch.tensor(
                masks, dtype=torch.float
            )
        else:
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample["image"]
            return torch.tensor([image], dtype=torch.float)
