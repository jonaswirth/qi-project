import h5py
import pandas as pd
import numpy as np
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d


# Load Galaxy10 dataset
def load_galaxy10(filepath):
    with h5py.File(filepath, "r") as f:
        ra = f["ra"][:]
        dec = f["dec"][:]
        labels = f["ans"][:]
        images = f["images"][:]
        rredshift = f["redshift"][:]
    galaxy10_data = pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "label": labels,
        "rredshift": rredshift,
        "image_idx": np.arange(0, len(images), dtype=int)  # Ensure integer indices
    })
    return galaxy10_data, images


# Match galaxies to SDSS spectra
def match_spectra(galaxy_data, radius=3 * u.arcsec):
    matches = []
    for _, row in galaxy_data.iterrows():
        coord = SkyCoord(ra=row["ra"] * u.deg, dec=row["dec"] * u.deg)
        result = SDSS.query_region(coord, radius=radius, spectro=True)
        if result is not None:
            matches.append({
                "ra": row["ra"],
                "dec": row["dec"],
                "label": row["label"],
                "rredshift": row["rredshift"],
                "image_idx": row["image_idx"],
                "redshift": result["z"][0] if "z" in result.colnames else None
            })
    return pd.DataFrame(matches)


# Fetch spectral flux data and resample to a common wavelength grid
def fetch_spectral_flux_parallel(data, batch_size=10, common_loglam=None):
    """
    Fetch spectral flux data for matched galaxies and resample to a common wavelength grid.

    Args:
        data (pd.DataFrame): Galaxy metadata including RA and Dec.
        batch_size (int): Number of galaxies to process in each batch.
        common_loglam (np.ndarray): Common logarithmic wavelength grid to resample to.

    Returns:
        pd.DataFrame: Filtered galaxy metadata with available spectra.
        dict: Mapping of indices to resampled flux arrays.
    """
    spectra_flux = {}
    filtered_data = []

    def fetch_spectrum(idx, row):
        """
        Fetch and resample spectrum for a single galaxy.

        Args:
            idx (int): Index of the galaxy.
            row (pd.Series): Metadata for the galaxy.

        Returns:
            tuple: (index, resampled flux array or None)
        """
        coord = SkyCoord(ra=row["ra"] * u.deg, dec=row["dec"] * u.deg)
        spec = SDSS.get_spectra(coordinates=coord)
        if spec and len(spec[0]) > 1:
            flux = spec[0][1].data["flux"]
            loglam = spec[0][1].data["loglam"]

            # Resample flux to the common wavelength grid
            interp = interp1d(loglam, flux, kind="linear", bounds_error=False, fill_value=0)
            resampled_flux = interp(common_loglam)

            return idx, resampled_flux
        return idx, None

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_idx = {
            executor.submit(fetch_spectrum, idx, row): idx
            for idx, row in data.iterrows()
        }

        for future in as_completed(future_to_idx):
            idx, flux = future.result()
            if flux is not None:
                spectra_flux[idx] = flux
                filtered_data.append(data.iloc[idx])

    # Return filtered data and resampled spectra
    return pd.DataFrame(filtered_data), spectra_flux


# Save combined dataset to HDF5
def save_to_hdf5(filepath, galaxy_data, images, spectra_flux):
    """
    Save Galaxy10 metadata, images, and resampled spectra as separate datasets in an HDF5 file.

    Args:
        filepath (str): Path to save the HDF5 file.
        galaxy_data (pd.DataFrame): Filtered galaxy metadata with redshifts and spectra.
        images (np.ndarray): Galaxy images.
        spectra_flux (dict): Mapping of indices to resampled flux arrays.
    """
    print("Saving dataset to HDF5...")

    # Create spectra array
    spectra_array = np.array([spectra_flux.get(int(row["image_idx"]), np.zeros(len(next(iter(spectra_flux.values())))))
                               for _, row in galaxy_data.iterrows()])

    # Save to HDF5
    with h5py.File(filepath, "w") as f:
        # Save metadata
        for col in galaxy_data.columns:
            f.create_dataset(col, data=galaxy_data[col].values)

        # Save images as a single dataset
        f.create_dataset("images", data=images)

        # Save spectra as a single dataset
        f.create_dataset("spectra", data=spectra_array)

    print(f"Data saved to {filepath}")


# Main script
if __name__ == "__main__":
    galaxy10_path = "../datasets/Galaxy10_DECals.h5"
    galaxy_data, images = load_galaxy10(galaxy10_path)
    print(f"Loaded Galaxy10 dataset with {len(galaxy_data)} galaxies.")

    # Limit to the first 10 galaxies for testing
    # galaxy_data = galaxy_data.iloc[:10]
    # images = images[:10]
    # print(f"Processing {len(galaxy_data)} galaxies for testing.")

    print("Cross-matching with SDSS spectra...")
    matched_data = match_spectra(galaxy_data)
    print(f"Matched {len(matched_data)} galaxies to SDSS spectra.")

    # Define a common wavelength grid
    common_loglam = np.linspace(3.5, 3.9, 4643)  # Example common grid (log10(λ))

    print("Fetching and resampling spectral flux data in parallel...")
    filtered_data, spectra_flux = fetch_spectral_flux_parallel(matched_data, batch_size=10, common_loglam=common_loglam)
    print(f"Spectra fetched and resampled for {len(filtered_data)} galaxies.")

    output_path = "../datasets/Galaxy10_with_resampled_spectra.h5"
    save_to_hdf5(output_path, filtered_data, images, spectra_flux)
