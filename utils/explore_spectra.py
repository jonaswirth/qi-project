import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../datasets/Galaxy10_with_spectra.h5"

with h5py.File(DATA_DIR, 'r') as f:
    print(f.keys())
    images = np.array(f["images"])
    redshift = np.array(f["redshift"])
    rredshift = np.array(f["rredshift"])
    spectra = np.array(f["spectra"])

    for i in range(0,10):
        print(f"Redshift: {redshift[i]} Rep. Redshift: {rredshift[i]} Image: {images[i].shape}, Spectrum: {spectra[i].shape}")
        # Plot image
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(images[i])
        plt.title(f"Galaxy {i} Image")

        # Plot spectrum
        plt.subplot(1, 2, 2)
        plt.plot(spectra[i])
        plt.title(f"Galaxy {i} Spectrum")
        plt.show()