import numpy as np

from PIL import Image
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


# Hongming's image_convert function to create pngs
def image_convert(name, image_data):
    """
    This function writes a PNG file from a numpy array.
    Args:
    name: Name of the output file without the .png suffix
    image_data: Input numpy array
    Returns:
    Writes PNG file to disk.
    Raises:
    KeyError: Raises an exception.
    """
    im = Image.fromarray(image_data)
    im = im.convert("L")
    im.save(name + ".png")
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
def read_fits_image(fitsfile, survey="VLA FIRST (1.4 GHz)", cropsize=150, pixel_arcsec=1.8):
    """
    This function extracts the image data from a FITS image, clips
    and linearly scales it.
    Args:
    fitsfile: Path to the input FITS file
    Returns:
    image_data: Numpy array containing image from FITS file
    Raises:
    KeyError: Raises an exception.
    """

    # Obtaining the naming convention
    namestring = fitsfile[18:52]

    with fits.open(fitsfile, ignore_missing_end=True) as hdu:
        image_data = hdu[0].data
        hdu.close()
    image_data = fits.getdata(fitsfile)

    #     # Check if the file has already been saved as a raw png
    #     if path.exists('./MiraBest'+extension+'/PNG/Raw/'+namestring+'_raw.png') is False:

    #         # Save for the first time here, as 'name_raw.png'
    #         image_convert('./MiraBest'+extension+'/PNG/Raw/'+namestring+'_raw', image_data)

    # set pixels < 3*noise to zero:
    a = sigma_clipped_stats(image_data, sigma=3.0, maxiters=5)
    image_data[np.where(image_data <= 3 * a[2])] = 0.0

    #     # Check if the file has already been saved as a clipped png
    #     if path.exists('./MiraBest'+extension+'/PNG/Clip/'+namestring+'_clip.png') is False:

    #         # Save for the second time here, as 'name_clip.png'
    #         image_convert('./MiraBest'+extension+'/PNG/Clip/'+namestring+'_clip', image_data)

    image_data = crop_centre(image_data, cropsize, cropsize)

    #     # Check if the file has already been saved as a cropped png
    #     if path.exists('./MiraBest'+extension+'/PNG/Crop/'+namestring+'_crop.png') is False:

    #         # Save for the third time here, as 'name_crop.png'
    #         image_convert('./MiraBest'+extension+'/PNG/Crop/'+namestring+'_crop', image_data)

    # Collect size data from name string
    source_size = float(namestring[27:40])  # Extent of radio source in arcsec
    pixel_size = np.ceil(
        source_size / pixel_arcsec
    )  # Converting to a size in pixels, rounded up to nearest pixel

    # normalise to [0, 1]:
    image_max, image_min = subset.max(), subset.min()
    image_data[mask] = (image_data[mask] - image_min) / (image_max - image_min)
    # remap to [0, 255] for greyscale:
    image_data *= 255.0

    # Check if the file has already been saved as the final png
    if path.exists("./MiraBest" + survey + "/PNG/Scaled_Final/" + namestring + ".png") is False:
        # Save for a final time here, as 'name.png'
        image_convert("./MiraBest" + survey + "/PNG/Scaled_Final/" + namestring, image_data)

    return image_data


if __name__ == "__main__":
