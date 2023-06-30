import requests
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
from urllib.request import urlretrieve

from utils import create_path


# Function to generate the appropriate filename convention for any entry
def name_string(entry):
    """This takes an entry with columns RA, dec, z, size_rad and class and makes a string to label it"""

    label = entry[4].astype(int).astype(str)
    ra = "{:07.3f}".format(entry[0] * 15)  # Moving both into degrees
    dec = "{:+08.3f}".format(entry[1])  # Retaining sign to keep length consistent
    z = "{:06.4f}".format(entry[2])  # All redshifts are < 0.5, so keep four significant figures
    rad = "{:07.2f}".format(entry[3])  # Radial size is a maximum of four figures before point

    name = label + "_" + ra + dec + "_" + z + "_" + rad

    return name


def image_download(entry, survey="VLA FIRST (1.4 GHz)", pixels=150):
    """Download an image from an entry of the same format as previously"""

    # Creating the path to the file and the name it'll be saved as
    dir = Path("MiraBest") / survey / "FITS"
    create_path(dir)
    filename = dir / (name_string(entry) + ".fits")

    ra, dec = entry[0], entry[1]

    # Preventing any duplicate downloads
    if filename.exists() is True:
        print(f"File {filename} already exists")
    else:
        coords = (ra, dec)
        # print(f"Downloading {filename}...")
        # coords = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        print(f"Attempting to query SkyView for image... (RA: {ra}, DEC: {dec})")
        location = SkyView.get_image_list(position=coords, survey=survey, pixels=pixels)
        print("Query successful!")

        try:
            # data = SkyView.get_images(position=coords, survey=["VLA FIRST (1.4 GHz)"])
            # Save fits file to disk
            # hdu = data[0][0]
            # hdu.writeto(filename, overwrite=True)

            print(f"Downloading image from {url[0]}")
            urlretrieve(location[0], filename)
            file = requests.get(url[0], allow_redirects=True)
            # print(filename)

            # Convert fits image to numpy array
            # hdu = fits.open(filename)[0]
            img = np.squeeze(hdu.data)
            # Plot image as sanity check
            plt.imshow(img, cmap="hot")
            plt.show()
            plt.savefig("fits_img.png")

        except:
            print("Problem with url", url)
            return None


# survey = "NVSS"  # ["VLA FIRST (1.4 GHz)", "NVSS"]
survey = "VLA FIRST (1.4 GHz)"


if __name__ == "__main__":
    with open("mirabestdata.txt", "r") as f:
        data = f.read().splitlines()

    dataset = []

    # Splitting out the relevant columns: in order, RA, dec, z, size_rad and FR_class
    for i in range(1, len(data)):
        columns = data[i].split()

        # A filter to remove any with radial extent greater than the image size
        if float(columns[7]) < 270:
            # A filter to remove any class 400 or 402 objects; these are "unclassifiable" and useless
            # for training
            if columns[8] != "400" and columns[8] != "402":
                if i == 1:
                    dataset = (np.asarray(columns[3:6] + columns[7:9])).astype(float)

                else:
                    columns = (np.asarray(columns[3:6] + columns[7:9])).astype(float)
                    dataset = np.concatenate((dataset, columns))

    # Final dataset with arrays of data for individual objects
    dataset = np.reshape(dataset, (-1, 5))

    # Download files from SkyView
    for i in range(len(dataset)):
        image_download(dataset[i], survey=survey, pixels=150)
