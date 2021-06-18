import os

import nibabel as nib
import torch
from torch.utils.data import Dataset

def get_path(dir):
    path_dict = {
        "dseg.json": [],
        "dseg.nii": [],
        "T2w.json": [],
        "T2w.nii": []
    }
    for i in range(1, 81):
        patient_path = os.path.join(
            dir, 
            "sub" + f"-{i:03}",
            "anat"
        )
        patient_files = os.listdir(patient_path)
        for patient_file in patient_files:
            for key in path_dict.keys():
                if key in patient_file:
                    path_dict[key].append(
                        os.path.join(patient_path, patient_file)
                    )
    return path_dict["T2w.nii"], path_dict["dseg.nii"]

class FeTA(Dataset):
    def __init__(
        self,
        data, 
        masks = None,
        augmentation = None
    ):
        super().__init__()
        self.data = data
        self.masks = masks
        self.augmentation = augmentation
        self.class_values = range(1, 8)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image  = self.data[item]

        if self.masks is not None:
            mask = self.masks[item]
            masks = [(mask == v).astype('uint8') for v in self.class_values]
            if self.augmentation:
                sample = self.augmentation(image=image, masks=masks)
                image, masks = sample['image'],  sample['masks']
            
            return torch.tensor([image], dtype=torch.float), \
                   torch.tensor(masks, dtype=torch.float)
        else:
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']
            return torch.tensor([image], dtype=torch.float)