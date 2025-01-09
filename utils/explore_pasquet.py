import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load("../datasets/data_example.npz")

# Inspect the keys (names of arrays in the file)
print("Keys:", data.keys())

# Access individual arrays
array1 = data["ebv"]  # Access the first unnamed array
array2 = data["z"]  # Access an array with a specific key
array3 = data["data"]
print("Array 1 shape:", array1.shape)
print("Array 2 shape:", array2.shape)
print("Array 3 shape:", array3.shape)

# Optionally convert to NumPy array if needed
array3 = np.array(array3)

first_image = array3[0]  # Shape: (64, 64, 5)

# Plot each channel separately
num_channels = first_image.shape[-1]
fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))

for i in range(num_channels):
    ax = axes[i]
    ax.imshow(first_image[:, :, i], cmap="gray", origin="lower")
    ax.set_title(f"Channel {i+1}")
    ax.axis("off")

plt.tight_layout()
plt.show()
