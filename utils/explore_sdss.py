import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../datasets/sdss_set.h5"

with h5py.File(DATA_DIR, 'r') as f:
    print(f.keys())
    images = np.array(f["images"])
    redshift = np.array(f["redshift"])
    spectra = np.array(f["spectra"])

    print(f"{len(images)} Images, {len(spectra)} Spectra, {len(redshift)} Redshifts")

    for i in range(0,10):
        print(f"Redshift: {redshift[i]} Image: {images[i].shape}, Spectrum: {spectra[i].shape}")

        image = images[i]
        g_band = image[1]  # g-band (slice 1)
        r_band = image[2]  # r-band (slice 2)
        i_band = image[3]

        image = np.stack([r_band, g_band, i_band], axis=-1)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Galaxy {i} Image")

        # Plot spectrum
        plt.subplot(1, 2, 2)
        plt.plot(spectra[i])
        plt.title(f"Galaxy {i} Spectrum")
        plt.show()