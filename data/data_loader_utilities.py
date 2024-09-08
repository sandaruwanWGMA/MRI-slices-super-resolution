import nibabel as nib
import numpy as np
import torch


def normalize_min_max(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def standardize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std()
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor


def load_nii_to_tensor(file_path, normalize="min_max"):
    img = nib.load(file_path)
    data = img.get_fdata()
    tensor = torch.from_numpy(data).float()

    # Apply normalization
    if normalize == "min_max":
        tensor = normalize_min_max(tensor)
    elif normalize == "standard":
        tensor = standardize_tensor(tensor)

    return tensor
