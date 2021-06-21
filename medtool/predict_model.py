import glob
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

import medtool.data.feta_dataset as feta_dataset


def get_mask(img):
    summed_up_masks = np.zeros((256, 256))
    for ind, i in enumerate(img):
        inversed_summed_up = ~((summed_up_masks).astype(bool))
        summed_up_masks += (i > 0.4) * inversed_summed_up * (ind + 1)

    return summed_up_masks


inputDir = "/input"
outputDir = "/output"

model = torch.load("/workspace/models/unetpp_se_resnext50_32x4d_balanced.pt")
model.to("cuda")
model.eval()

T2wImagePath = glob.glob(os.path.join(inputDir, "anat", "sub-*_T2w.nii.gz"))[0]
nifty_image = nib.load(T2wImagePath)
sub = os.path.split(T2wImagePath)[1].split("_")[0]

images = np.array(np.moveaxis(np.asarray(nifty_image.dataobj), -1, 0)).reshape(
    -1, 256, 256
)

test_dataset = feta_dataset.FeTA(images)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)

result = []
for one_slice in test_loader:
    one_slice = one_slice.to("cuda")
    output = model(one_slice)[0]

    stacked_mask = get_mask(output.detach().cpu().numpy())
    result.append(stacked_mask)

result = np.moveaxis(np.array(result), 0, -1)

nib.save(
    nib.Nifti1Image(result, nifty_image.affine, nifty_image.header),
    os.path.join(outputDir, sub + "_seg_result.nii.gz"),
)
