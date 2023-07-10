import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.neighbors import NearestNeighbors

from utils import create_path


# Function to generate the appropriate filename convention for any entry


def read_metadata(filename="mirabestdata.txt"):
    # Read the data from the file into a DataFrame
    data = pd.read_csv("mirabestdata.txt", delim_whitespace=True, skiprows=1, header=None)

    # Remove columns 0-3 and 6
    data = data.drop([0, 1, 2, 6], axis=1)

    # Rename the columns for readability
    data.columns = ["ra", "dec", "z", "size", "label"]

    # Apply filters to remove rows with radial extent greater than 270 and class 400 or 402 objects
    data = data[(data["size"] < 270) & (~data["label"].isin(["400", "402"]))]

    # Convert from decimal hours to degrees
    data["ra"] = data["ra"] * 15

    # Reset index to get consecutive integers as the index
    data.reset_index(drop=True, inplace=True)

    return data


def update_metadata(mb_metadata, rgz_metadata, threshold=15):
    """
    For each source in in the MiraBest data, find the nearest source in the RGZ data and
    replace the size in the MiraBest data wtih the corresponding size from the RGZ data-set
    """

    # Fit sklearn nearest neighbours model to the RGZ data
    rgz_coords = rgz_metadata[["radio.ra", "radio.dec"]].values
    rgz_nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(rgz_coords)

    # Find the nearest RGZ source to each MiraBest source
    mb_coords = mb_metadata[["ra", "dec"]].values
    distances, indices = rgz_nn.kneighbors(mb_coords)

    print(
        f"""Number of MiraBest sources with angular size replaced by RGZ DR1 value: 
        {np.sum(distances < threshold)} of {len(mb_metadata)} (threshold: {threshold} arcsec) \n"""
    )

    # Replace the size in the MiraBest data with the corresponding size from the RGZ data
    #  if distance is less than 0.1 arcsec
    mb_metadata["size"] = np.where(
        distances < threshold,
        rgz_metadata["radio.max_angular_extent"].values[indices],
        mb_metadata["size"],
    )

    return mb_metadata


def name_string(label, ra, dec, z, size):
    """This takes an entry with columns RA, dec, z, size_rad and class and makes a string to label it"""

    label = label.astype(int).astype(str)
    ra = "{:07.3f}".format(ra)  # Moving both into degrees
    dec = "{:+08.3f}".format(dec)  # Retaining sign to keep length consistent
    z = "{:06.4f}".format(z)  # All redshifts are < 0.5, so keep four significant figures
    size = "{:07.2f}".format(size)  # Radial size is a maximum of four figures before point

    name = label + "_" + ra + dec + "_" + z + "_" + size

    return name


def image_download(label, ra, dec, z, size, survey="VLA FIRST (1.4 GHz)", pixels=150):
    """Download an image from an entry of the same format as previously"""

    # Creating the path to the file and the name it'll be saved as
    dir = Path("MiraBest") / survey / "FITS"
    create_path(dir)
    filename = dir / (name_string(label, ra, dec, z, size) + ".fits")

    # print(f"Downloading {filename}...")
    print(f"Attempting to query SkyView for image... (RA: {ra}, DEC: {dec})")
    coords = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    # try:
    #     hdu = SkyView.get_images(position=coords, survey=["VLA FIRST (1.4 GHz)"])[0]

    # except:
    #     print("Failed to query SkyView, most likely timeout.")
    #     return

    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    url = SkyView.get_image_list(position=sky, survey=survey, cache=False, pixels=pixels)
    try:
        file = requests.get(url[0], allow_redirects=True)
    except:
        print("Unable to download", filename)
        return None

    try:
        open(filename, "wb").write(file.content)
    except:
        print("No FITS available:", filename)
        return None

    hdu = fits.open(filename)
    # hdu.writeto(filename, overwrite=True)

    img = np.squeeze(hdu[0].data)
    # Plot image as sanity check
    plt.imshow(img, cmap="hot")
    plt.savefig("fits_img.png")
    plt.close()
    print("Successfully pulled image from SkyView")


if __name__ == "__main__":
    # survey = "NVSS"  # ["VLA FIRST (1.4 GHz)", "NVSS"]
    survey = "VLA FIRST (1.4 GHz)"

    # Download files from SkyView
    meta_data = read_metadata()
    rgz_metadata = pd.read_csv("rgz.csv")
    meta_data = update_metadata(meta_data, rgz_metadata)

    # Save dataframe as .parquet
    meta_data.to_parquet("mirabest.parquet")

    for i, row in meta_data.iterrows():
        image_download(
            row["label"], row["ra"], row["dec"], row["z"], row["size"], survey=survey, pixels=150
        )
