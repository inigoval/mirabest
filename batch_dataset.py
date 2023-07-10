import numpy as np
import os.path as path
import random
import pickle
import io

from pathlib import Path
from PIL import Image

from utils import create_path, load_config


def str_to_label(string):
    """Converts 3 digit string to label. Sets labels outside of desired labels to None"""

    hashmap = {
        "100": 0,
        "102": 1,
        "104": 2,
        "110": 3,
        "112": 4,
        "200": 5,
        "201": 6,
        "210": 7,
        "300": 8,
        "310": 9,
    }

    return hashmap.get(string, None)


def filename_to_label(filename):
    label_string = str(filename)[-38:-35]
    label = str_to_label(label_string)
    return label


def filename_to_img(filename):
    img = Image.open(filename)
    img = np.array(img)
    img = np.array(list(img), np.uint8)

    return img


def build_dataset(path, n_batches=7):
    batch_dir = path / "batches"
    create_path(batch_dir)
    png_dir = path / "PNG"

    # All filenames in png directory
    filenames = sorted(list(png_dir.glob("*.png")))

    # Filter out classes 40x (unclassifiable) and 103 (diffuse FRI)
    filenames = [filename for filename in filenames if filename_to_label(filename) is not None]

    # Total number of pixels in each image
    nvis = np.prod(filename_to_img(filenames[0]).shape)

    print(f"Number of images (post filtering):{len(filenames)} \n")

    with open("test_names.pkl", "rb") as f:
        test_names = pickle.load(f)

    batches = []

    # Test batch
    batch = {"labels": [], "data": [], "filenames": [], "batch_label": "testing batch 1 of 1"}
    for test_name in test_names:
        test_name = list(png_dir.glob(f"{test_name[-38:-17]}*.png"))
        assert len(test_name) == 1
        test_name = test_name[0]
        batch["labels"].append(filename_to_label(test_name))
        batch["data"].append(filename_to_img(test_name))
        batch["filenames"].append(str(test_name))
        filenames.remove(test_name)

    with io.open(batch_dir / "test_batch", "wb") as f:
        pickle.dump(batch, f)

    print(f"Test batch containing {len(batch['labels'])} images saved\n")
    print(f"Number of images remaining: {len(filenames)}\n")

    # Training batches
    n_batches = 7
    batch_size = len(filenames) // (n_batches - 1)

    print(f"Using a batch size of {batch_size} for a total of {n_batches} batches\n")

    # Seed batching
    random.seed(42)
    for i in range(n_batches):
        batch = {}
        batch = {
            "labels": [],
            "data": [],
            "filenames": [],
            "batch_label": f"training batch {i+1} of {n_batches}",
        }

        for j in range(min(batch_size, len(filenames))):
            filename = random.choice(filenames)
            batch["labels"].append(filename_to_label(filename))
            batch["data"].append(filename_to_img(filename))
            batch["filenames"].append(str(filename))
            filenames.remove(filename)

        with io.open(batch_dir / f"data_batch_{i+1}", "wb") as f:
            pickle.dump(batch, f)

        print(f"Batch {i+1} containing {len(batch['labels'])} images saved")

    # create dictionary of batch:
    metadata = {
        "num_cases_per_batch": batch_size,
        "label_names": ["100", "102", "104", "110", "112", "200", "201", "210", "300", "310"],
        "num_vis": nvis,
    }

    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    with io.open(batch_dir / "batches.meta", "wb") as f:
        pickle.dump(dict, f)


if __name__ == "__main__":
    config = load_config()
    path = Path("MiraBest") / config["survey"]

    build_dataset(path, n_batches=config["n_batches"])
