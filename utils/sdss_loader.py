import matplotlib.pyplot as plt
import numpy as np
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
from skimage.transform import resize

def fetch_all_bands(ra, dec, radius=5 * u.arcsec, image_size=512):
    """
    Fetch all image bands (u, g, r, i, z) for a given RA and Dec.
    Args:
        ra (float): Right Ascension of the galaxy.
        dec (float): Declination of the galaxy.
        radius (astropy.units.Quantity): Radius for the image cutout.
        image_size (int): Size to resize images (image_size x image_size).

    Returns:
        dict: Dictionary with keys as bands ('u', 'g', 'r', 'i', 'z') and resized image data as values.
    """
    bands = ["u", "g", "r", "i", "z"]
    images = {}
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    xid = SDSS.query_region(coord, radius=radius, spectro=True)
    print(xid)

    for band in bands:
        try:
            image = SDSS.get_images(matches=xid, band=band)
            if not image:
                print(f"No image available for band {band}.")
                images[band] = None
            else:
                print(image[0][0].data.shape)
                image_data = image[0][0].data
                images[band] = image_data
        except Exception as e:
            print(f"Error fetching band {band}: {e}")
            images[band] = None

    return images

def display_visible_bands(images):
    """
    Display all individual bands (u, g, r, i, z) from the given dictionary.
    Args:
        images (dict): Dictionary of images with bands as keys ('u', 'g', 'r', 'i', 'z').
    """
    bands = ["u", "g", "r", "i", "z"]
    num_bands = len(bands)

    # Create a figure with subplots for each band
    fig, axes = plt.subplots(1, num_bands, figsize=(15, 5))

    for i, band in enumerate(bands):
        ax = axes[i]
        if images[band] is not None:
            ax.imshow(images[band], cmap="gray", origin="lower")
            ax.set_title(f"Band: {band.upper()}")
        else:
            ax.set_title(f"Band: {band.upper()} (Unavailable)")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
def main():
    # Refined SQL query to ensure bright, low-redshift galaxies are selected
    query = """
    SELECT TOP 1 
        po.ra, po.dec, za.z 
    FROM SpecObjAll za
    JOIN PhotoObjAll po ON (po.objID = za.bestObjID)
    WHERE 
        za.z > 0 AND za.zWarning = 0
        AND za.TargetType = 'SCIENCE' and za.survey = 'sdss'
        AND za.class = 'GALAXY' and za.primtarget >= 64
        AND (po.petroMag_r - po.extinction_r) >= 17.8  -- Bright galaxies
        AND po.clean = 1 AND po.insideMask = 0
    """
    result = SDSS.query_sql(query)

    if result is None or len(result) == 0:
        print("No galaxies found.")
        return
    
    for i in range(0, len(result)):
        # Extract relevant fields for the first galaxy
        galaxy = result[i]
        ra = galaxy["ra"]
        dec = galaxy["dec"]

        #Ra: 331.66405522483547, Dec: -0.4841546865670835
        ra = 331.66405522483547
        dec = -0.4841546865670835

        print(f"Fetching data for Galaxy: RA={ra}, Dec={dec}")

        # Fetch all bands
        images = fetch_all_bands(ra, dec)
        if not images:
            print("No images fetched.")
            return

        # Display visible bands
        display_visible_bands(images)

if __name__ == "__main__":
    main()
