import os

from torch.utils.data import Dataset

def get_path(dir):
    path_dict = {
        'dseg.json': [],
        'dseg.nii': [],
        'T2w.json': [],
        'T2w.nii': []
    }
    for i in range(1, 81):
        patient_path = os.path.join(
            dir, 
            "sub" + f"-{i:03}",
            "anat"
        )
        patient_files = os.listdir(patient_path)
        for p in patient_files:
            for k in path_dict.keys():
                if k in p:
                    path_dict[k].append(p)
    return path_dict

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
        image  = self.data[item]
        
        if self.labels:
            labels = self.labels[item]
            return {
                'image': image,
                'labels': labels 
            }
        else:
            return {
                'image': image
            }