# Ca2+-Image-Utils

## Overview

Ca2+-Image-Utils is a Python utility for processing and comparing neurofinder test datasets, specifically designed for calcium imaging analysis. The project provides tools to convert coordinates to masks, visualize image comparisons, and perform dataset evaluations.

## Features

- Convert coordinate-based ROIs to boolean masks
- Compare ground truth and predicted neuron regions
- Visualize image comparisons across multiple datasets
- Command-line interface for easy usage

## Usage

### Command-Line Interface

```bash
python plot_comparison.py -i /path/to/input/directory -pred /path/to/predictions.json [--save_figs]
```

#### Arguments
- `-i, --input_dir`: Directory containing neurofinder datasets
- `-pred`: JSON file with prediction results
- `--save_figs`: Optional flag to save comparison images

### Example

```bash
python plot_comparison.py -i ./neurofinder_datasets -pred ./results.json --save_figs
```

## Main Functions

- `coordinates_to_mask()`: Converts coordinate arrays to boolean masks
- `plot_comparison_image()`: Creates visual comparisons between ground truth and predicted neuron regions
- `main()`: Processes multiple datasets and generates comparisons


## Further reading
[Visit](futherreading.md)

![Bild1](resources/Pic1.png)