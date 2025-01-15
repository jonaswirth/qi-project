#Loads data from the original astroclip dataset and stores more manageable chuncks
import h5py
import matplotlib.pyplot as plt
import numpy as np

def crop_image(img, cutout_size):
    height, width, _ = img.shape

    center_x, center_y = height // 2, width // 2
    from_y = center_y - cutout_size // 2
    to_y = center_y + cutout_size // 2
    from_x = center_x - cutout_size // 2
    to_x = center_x + cutout_size //2

    return img[from_y:to_y, from_x:to_x, :]

def process_images(images, cutout_size):
    processed_images = []
    
    for img in images:
        # Crop the image
        cropped_img = crop_image(img, cutout_size)
        
        # Rearrange channels from (g, r, z) to (r, g, b)
        cropped_img = cropped_img[:, :, [1, 0, 2]]
        
        processed_images.append(cropped_img)
    
    # Convert list back to NumPy array
    return np.array(processed_images)

def read_from_file(k, n_samples):
    with h5py.File("../datasets/astroclip_reduced_2.h5", "r") as f:
        images = f["images"][:n_samples]
        spectra = f["spectra"][:n_samples]
        redshifts = f["redshifts"][:n_samples]

        return images, spectra, redshifts

def store_file(images, spectra, redshifts):
    with h5py.File("../datasets/astroclip_reduced_3.h5", "w") as f:
        f.create_dataset('images', data=images, compression="gzip", compression_opts=9)
        f.create_dataset('spectra', data = spectra)
        f.create_dataset('redshifts', data=redshifts)

if __name__ == "__main__":
    images, spectra, redshifts = read_from_file("0", 20000)
    images = process_images(images, 64)
    store_file(images, spectra, redshifts)
