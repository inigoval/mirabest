import numpy as np

from PIL import Image
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from utils import create_path


# Hongming's image_convert function to create pngs
def image_convert(savepath, img):
    """
    This function writes a PNG file from a numpy array.
    Args:
    name: Name of the output file without the .png suffix
    img: Input numpy array
    Returns:
    Writes PNG file to disk.
    Raises:
    KeyError: Raises an exception.
    """
    im = Image.fromarray(img)
    im = im.convert("L")
    im.save(savepath)
    return


# A modified version of Hongming's crop_center function
def crop_centre(img, cropx, cropy):
    """ "
    This function crop images from centre to given size.
    Args:
    img: input image
    cropx: output image width
    cropy: output image height
    Returns:
    data of cropped img
    Raises:
    """

    xsize = np.shape(img)[0]  # image width
    ysize = np.shape(img)[1]  # image height
    startx = xsize // 2 - (cropx // 2)
    starty = ysize // 2 - (cropy // 2)
    img_slice = img[starty : starty + cropy, startx : startx + cropx]
    # This is a sub-optimal solution
    return img_slice


# A modified version of Hongming's read_fits_image function; now creates all required pngs
def read_fits_image(fitsfile, survey="VLA FIRST (1.4 GHz)"):
    """
    This function extracts the image data from a FITS image, clips
    and linearly scales it.

    Args:
        fitsfile: Path to the input FITS file
        survey: Name of the survey the FITS file is from

    Returns:
        img: Numpy array containing image from FITS file
    """

    dir = Path("MiraBest") / survey / "PNG"
    create_path(dir)

    # Obtaining the naming convention
    namestring = str(fitsfile)[-39:-5]

    # with fits.open(fitsfile, ignore_missing_end=True) as hdu:
    #     img = hdu[0].data
    #     hdu.close()

    img = fits.getdata(fitsfile)

    # Remove nans
    img[np.where(np.isnan(img))] = 0.0

    # Sigma clipping
    _, _, rms = sigma_clipped_stats(img)
    img[np.where(img <= 3 * rms)] = 0.0

    # normalise to [0, 1]:
    image_max, image_min = img.max(), img.min()
    img = (img - image_min) / (image_max - image_min)
    # remap to [0, 255] for greyscale:
    img *= 255.0

    # Check if the file has already been saved as the final png
    savepath = dir / (namestring + ".png")
    image_convert(str(savepath), img)

    return img


if __name__ == "__main__":
    config = load_config()
    dir = Path("MiraBest") / config["survey"] / "FITS"

    list = dir.glob("*.fits")

    # List of files confirmed to have missing data
    blacklist = [
        "100_036.061+000.563_0.1400_0093.07",
        "200_003.198+000.788_0.1484_0026.82",
        "200_033.655+000.710_0.2900_0030.76",
        "210_348.925-000.435_0.0910_0050.37",
    ]

    for file in list:
        # Create images for any file not in the blacklist
        if str(file)[18:52] not in blacklist:
            print(f"Saving png file: {str(file)}")
            read_fits_image(file, survey=config["survey"])
