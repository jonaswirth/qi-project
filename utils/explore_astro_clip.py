import h5py
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "../datasets/astroclip_reduced_3.h5"
img1_idx = 1
img2_idx = 5



with h5py.File(FILE_PATH, "r") as f:
    img1 = np.array(f["images"][img1_idx])
    img2 = np.array(f["images"][img2_idx])
    spec1 = np.array(f["spectra"][img1_idx])
    spec2 = np.array(f["spectra"][img2_idx])
    redshift1 = np.array(f["redshifts"][img1_idx])
    redshift2 = np.array(f["redshifts"][img2_idx])


#img1[:, :, 2] = img1[:, :, 2] / np.max(img1[:, :, 2]) * np.max(img1[:, :, [0, 1]])


#img2[:, :, 2] = img2[:, :, 2] / np.max(img2[:, :, 2]) * np.max(img2[:, :, [0, 1]])

# Create the plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Group 1 (img1 and spec1)
axes[0, 0].imshow(img1)
axes[0, 0].axis("off")  # Turn off axis for image
axes[0, 1].plot(spec1)
axes[0, 1].set_title(f"Redshift: {redshift1:.3f}")  # Title for Group 1

# Group 2 (img2 and spec2)
axes[1, 0].imshow(img2)
axes[1, 0].axis("off")  # Turn off axis for image
axes[1, 1].plot(spec2)
axes[1, 1].set_title(f"Redshift: {redshift2:.3f}")  # Title for Group 2

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
