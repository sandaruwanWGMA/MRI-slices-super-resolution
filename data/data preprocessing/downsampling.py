import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import os

base_directory = os.getcwd()
current_directory = os.path.abspath(os.path.join(base_directory, os.pardir))
file_name = "sub-HC001_ses-01_acq-mp2rage_T1map.nii.gz"

file_path = os.path.join(
    current_directory, "Sample Data/sub-HC001_ses-01_acq-mp2rage_T1map.nii.gz"
)

# Check if the file exists at the given path
if os.path.exists(file_path):
    # Open and read the file
    with open(file_path, "r") as file:
        nib_file = nib.load(file_path)
        print("File loaded sucessfully")
else:
    print(f"File does not exist at path: {file_path}")

header = nib_file.header

# Downsample the slices by selecting every 5th slice
downsampled_data = nib_file.slicer[::5, :, :]

output_path = os.path.join(
    current_directory, f"Sample Data/downsampled_and_without_rotations_{file_name}"
)
nib.save(downsampled_data, output_path)

downsampled_file = nib.load(output_path)

downsampled_header = downsampled_data.header

header_df = pd.DataFrame(
    {
        "Header Field": header.keys(),
        "Original Header": header.values(),
        "Downsampled Header": downsampled_header.values(),
    }
)

header_df.to_csv(
    "../Results/After removing rotations and inccorect affine matrix of downsampled data/original_vs_sampled_header_params.csv",
    index=False,
)
header_df.to_excel(
    "../Results/After removing rotations and inccorect affine matrix of downsampled data/original_vs_sampled_header_params.xlsx",
    index=False,
)


def show_slices(slices):
    """Function to display row of image slices"""
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


original_shape = nib_file.shape
nib_data = nib_file.get_fdata()
slice_0 = nib_data[original_shape[0] // 2, :, :]
slice_1 = nib_data[:, original_shape[1] // 2, :]
slice_2 = nib_data[:, :, original_shape[2] // 2]
show_slices([slice_0, slice_1, slice_2])

plt.suptitle("Center slices for original image")

sampled_shape = downsampled_data.shape
sampled_data = downsampled_data.get_fdata()
slice_0 = sampled_data[sampled_shape[0] // 2, :, :]
slice_1 = sampled_data[:, sampled_shape[1] // 2, :]
slice_2 = sampled_data[:, :, sampled_shape[2] // 2]
show_slices([slice_0, slice_1, slice_2])

plt.suptitle("Center slices for sampled image")
