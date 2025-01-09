import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../datasets/Galaxy10_DECals.h5"

with h5py.File(DATA_DIR, 'r') as f:
    ra = np.array(f["ra"][0])
    dec = np.array(f["dec"][0])
    image = np.array(f["images"][0])

    print(f"Ra: {ra}, Dec: {dec}")
    plt.imshow(image)
    plt.show()

    classes = np.array(f['ans'])
    labels = np.array(f['redshift'])

    valid_indices = ~np.isnan(labels)
    classes = classes[valid_indices]
    labels = labels[valid_indices]

    for c in range(0, 9):
        class_idx = np.where(classes == c)
        class_labels = labels[class_idx]

        plt.subplot(3,3,c+1)
        plt.boxplot(class_labels)
        plt.title(f"Class {c}")
        plt.xticks([1], [''])
    
    plt.show()