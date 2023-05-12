"""Plot comparison between neurofinder test-datasets and a results.json

This script allows the user to compare the results to the neurofinder
test data.

This file can also be imported as a module and contains the following
functions:

    * coordinates_to_mask - converts coordinates to a boolean mask
    * plot_comparison_image - creates the comparison between GT and PRED
    * main - the main function of the script
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from neurofinder_to_hdf5 import image_dir_to_array
from numpy import array, ndarray, zeros


def coordinates_to_mask(coords: ndarray, dims: Tuple[int, int]) -> ndarray:
    """Convert array of coordinates of rois to a boolean mask

    Parameters
    ----------
    coords : ndarray
        Array of coordinates with the shape [2, X]
    dims : Tuple[int, int]
        Dimension of the image the mask is supposed to represent

    Returns
    -------
    ndarray
        Boolean mask of the converted coordinates
    """
    mask = zeros(dims)
    mask[tuple(array(coords).T)] = 1
    return mask


def plot_comparison_image(args, ds_name, imgs, gt_masks, pred_masks):
    """Plot the comparison image"""
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(imgs.sum(axis=0), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(gt_masks.sum(axis=0), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_masks.sum(axis=0), cmap="gray")
    plt.show()

    if args.save_figs:
        plt.savefig(
            args.pred.with_name(f"comp_{ds_name}-{args.pred.parent.name}.png")
        )


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "input_dir",
        type=Path,
        help=(
            "Directory of one or multiple neurofinder datasets for testing."
            "Must include the results.json-File."
        ),
    )
    parser.add_argument(
        "-pred",
        type=Path,
        help="Second json-File to compare with",
    )
    parser.add_argument(
        "--save_figs",
        action=argparse.BooleanOptionalAction,
        help="Optional, whether to save the comparison images.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_cli_args()

    image_dirs: List[Path] = [
        Path(dir) for dir in os.walk(args.input_dir)[0] if "images" in dir
    ]
    gt_results: List[Path] = [
        (dir.parent / "result.json") for dir in image_dirs
    ]
    dataset_names: List[str] = [d.parent.name for d in image_dirs]

    for im_dir, gt_file, ds_name in zip(image_dirs, gt_results, dataset_names):
        imgs = image_dir_to_array(im_dir)
        dims = imgs.shape[1:]

        with open(gt_file, "r", encoding="utf-8") as gt_file:
            gt_regions = json.load(gt_file)[0]["regions"]
        with open(args.pred, "r", encoding="utf-8") as pred_file:
            pred_regions = json.load(pred_file)

        gt_masks = array(
            [coordinates_to_mask(s["coordinates"], dims) for s in gt_regions]
        )
        pred_masks = array(
            [coordinates_to_mask(s["coordinates"], dims) for s in pred_regions]
        )

        plot_comparison_image(args, ds_name, imgs, gt_masks, pred_masks)


if __name__ == "__main__":
    main()
