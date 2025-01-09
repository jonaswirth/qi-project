#Loads data from the original astroclip dataset and stores more manageable chuncks
import h5py
import numpy as np

def read_from_file(n_samples):
    with h5py.File("../datasets/astroclip_reduced_1.h5", "r") as f:
        images = f["images"][:n_samples]
        spectra = f["spectra"][:n_samples]
        redshifts = f["redshifts"][:n_samples]

        in_range = np.where(redshifts < 0.5)

        images = images[in_range]
        spectra = spectra[in_range]
        redshifts = redshifts[in_range]

        print(f"Images: {images.shape}, Spectra {spectra.shape}, Redshifts: {redshifts.shape}")

        return images, spectra, redshifts

def store_file(images, spectra, redshifts):
    with h5py.File("../datasets/astroclip_reduced_2.h5", "w") as f:
        f.create_dataset('images', data=images, compression="gzip", compression_opts=9)
        f.create_dataset('spectra', data = spectra)
        f.create_dataset('redshifts', data=redshifts)

if __name__ == "__main__":
    images, spectra, redshifts = read_from_file(500)
    store_file(images, spectra, redshifts)