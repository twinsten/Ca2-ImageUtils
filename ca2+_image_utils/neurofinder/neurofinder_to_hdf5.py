"""Convert Neurofinder folder(s) to hdf5-dataset(s)

This script allows the user to convert the image folder of one or more
neurofinder datasets to separate hdf5-datasets. By default the dataset
inside the hdf5-dataset will be an array of shape [x,y,images]

This script requires that `skimage` and `h5py` be installed within the
Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * image_dir_to_array - converts an directory of images to an array
    * write_hdf5_dataset - writes a stack of images to a hdf5-file
    * main - the main function of the script
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
from skimage import io


def write_hdf5_dataset(
    im_array: np.ndarray, hdf5_path: Path, dataset_name: str = "video"
) -> None:
    """Writes an array of image data to a hdf5-file.

    Parameters
    ----------
    im_array : np.ndarray
        Input array of image data in the format [x,y,num_images]
    hdf5_path : Path
        Path where the hdf5-file will be saved
    dataset_name : str, optional
        Name of the dataset inside the hdf5-file, by default "video"
    """
    with h5py.File(hdf5_path.with_suffix(".h5"), "w") as h5f:
        h5f.create_dataset(dataset_name, data=im_array)

        logging.info(f"Saved dataset to: {hdf5_path.with_suffix('.h5')}")


def image_dir_to_array(
    im_dir: Path, extension: Union[str, List[str]] = ".png"
) -> np.ndarray:
    """Iterate a directory and load images.

    Parameters
    ----------
    im_dir : Path
        Path to the input directory
    extension : Union[str, List[str]]
        Extension(s) to search for, by default ".png"

    Returns
    -------
    np.ndarray
        Array of image data in the format [x,y,num_images]
    """
    imgs: List[np.ndarray] = []
    for im_path in im_dir.iterdir():
        if im_path.suffix == extension:
            im = np.array(io.imread(im_path, as_gray=True))
            imgs.append(im)

    return np.array(imgs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        help="The input directory of one or multiple neurofinder datasets",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="The output directory of the hdf5-datasets",
    )
    parser.add_argument(
        "--image_extension",
        type=str,
        default=".png",
        help="The extension of the image files",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="video",
        help="The name of the dataset inside each hdf5-file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"Working directory is: {os.getcwd()}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_dirs: List[Path] = [
        Path(dir) for dir in os.walk(args.input_dir)[0] if "images" in dir
    ]
    dataset_names: List[str] = [d.parent.name for d in image_dirs]

    logging.info(f"No. of datasets found: {len(image_dirs)}")

    for im_dir, ds_name in zip(image_dirs, dataset_names):
        imgs = image_dir_to_array(im_dir, args.image_extension)
        logging.debug(f"Shape of image-stack: {imgs.shape}")

        write_hdf5_dataset(
            imgs,
            args.output_dir / (ds_name + args.image_extension),
            dataset_name=args.dataset_name,
        )

    logging.info("Successfully saved all datasets!")


if __name__ == "__main__":
    main()
