import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

import medtool.data.mnm_dataset as mnm_dataset

outputDir = "./Submission"

model = torch.load("./models/mnm_unetpp_se_resnext50_32x4d.pt")
model.to("cuda")
model.eval()


def get_mask(img, original_shape):
    summed_up_masks = np.zeros(img[0].shape)
    for ind, i in enumerate(img):
        inversed_summed_up = ~((summed_up_masks).astype(bool))
        summed_up_masks += (i > 0.4) * inversed_summed_up * (ind + 1)
    res = np.zeros(original_shape)
    if original_shape[0] < 256 or original_shape[1] < 256:
        res = summed_up_masks[: original_shape[0], : original_shape[1]]
    else:
        res[: summed_up_masks.shape[0], : summed_up_masks.shape[1]] = summed_up_masks
    return res


def pad_with_zeros(img):
    if img.shape[-1] < 256:
        zeros = torch.zeros((1, 1, img.shape[-2], 256))
        zeros[0][0][: img.shape[-2], : img.shape[-1]] = img
        img = zeros
    if img.shape[-2] < 256:
        zeros = torch.zeros((1, 1, 256, img.shape[-1]))
        zeros[0][0][: img.shape[-2], : img.shape[-1]] = img
        img = zeros
    return img


for sub in range(161, 201):
    for ls in ["LA", "SA"]:
        for ds in ["ED", "ES"]:
            print(sub, ls, ds)
            nifty_image = nib.load(
                f"../MnM-2/validation/{sub:03}/{sub:03}_{ls}_{ds}.nii.gz"
            )
            data = np.moveaxis(nifty_image.get_fdata(), -1, 0)
            # print(data.shape)
            test_dataset = mnm_dataset.MnM(data)

            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                drop_last=False,
            )

            result = []
            for one_slice in test_loader:
                original_shape = np.array(one_slice[0][0].shape)
                one_slice = pad_with_zeros(one_slice)
                one_slice = one_slice[:, :, :320, :320].to("cuda")
                output = model(one_slice)[0]

                stacked_mask = get_mask(output.detach().cpu().numpy(), original_shape)
                result.append(stacked_mask)

            result = np.moveaxis(np.array(result), 0, -1)
            directory = os.path.join(outputDir, str(sub))

            if not os.path.exists(directory):
                os.makedirs(directory)
            nib.save(
                nib.Nifti1Image(result, nifty_image.affine, nifty_image.header),
                os.path.join(directory, f"{sub}_{ls}_{ds}_pred.nii.gz"),
            )
