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

from utils import create_path, load_config
from download_images import image_download, read_metadata, update_metadata
from fits_to_png import read_fits_image
from batch_dataset import build_dataset

if __name__ == "__main__":
    config = load_config()

    # Download files from SkyView
    meta_data = read_metadata()
    rgz_metadata = pd.read_csv("rgz.csv")
    meta_data = update_metadata(meta_data, rgz_metadata)

    # Save dataframe as .parquet
    meta_data.to_parquet("mirabest.parquet")

    for i, row in meta_data.iterrows():
        image_download(
            row["label"],
            row["ra"],
            row["dec"],
            row["z"],
            row["size"],
            survey=config["survey"],
            pixels=config["crop_size"],
        )

    print("FITS downloads completed\n")
    print("Converting FITS files to PNG images...\n")

    # Main directory for data extraction
    dir = Path("MiraBest") / config["survey"]
    create_path(dir, dir / "FITS", dir / "PNG")

    # List of files confirmed to have missing data
    blacklist = [
        "100_036.061+000.563_0.1400_0093.07",
        "200_003.198+000.788_0.1484_0026.82",
        "200_033.655+000.710_0.2900_0030.76",
        "210_348.925-000.435_0.0910_0050.37",
    ]

    # Convert all FITS files to PNG
    for file in (dir / "FITS").glob("*.fits"):
        # Create images for any file not in the blacklist
        if str(file)[18:52] not in blacklist:
            print(f"Saving png file: {str(file)}")
            read_fits_image(file, survey=config["survey"])

    print("PNG conversion completed\n")
    print("Batching dataset...\n")

    build_dataset(dir, n_batches=config["n_batches"])
