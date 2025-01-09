#Loads data from the original astroclip dataset and stores more manageable chuncks
import h5py
import matplotlib.pyplot as plt
import numpy as np

def read_from_file(k, n_samples):
    with h5py.File("../datasets/astroclip_desi.1.1.5.h5", "r") as f:
        images = f[k]["images"][:n_samples]
        spectra = f[k]["spectra"][:n_samples]
        redshifts = f[k]["redshifts"][:n_samples]

        return images, spectra, redshifts

def store_file(images, spectra, redshifts):
    with h5py.File("../datasets/astroclip_reduced_1.h5", "w") as f:
        f.create_dataset('images', data=images, compression="gzip", compression_opts=9)
        f.create_dataset('spectra', data = spectra)
        f.create_dataset('redshifts', data=redshifts)

if __name__ == "__main__":
    images, spectra, redshifts = read_from_file("0", 20000)
    store_file(images, spectra, redshifts)
