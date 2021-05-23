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
        labels = None
    ):
        super().__init__()
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path  = self.data[item]
        image = nib.load(image_path).get_fdata()

        if self.labels:
            labels_path = self.labels[item]
            labels = nib.load(labels_path).get_fdata()
            return {
                'image':  torch.tensor(image, dtype=torch.float),
                'labels': torch.tensor(labels, dtype=torch.float)
            }
        else:
            return {
                'image':  torch.tensor(image, dtype=torch.float)
            }