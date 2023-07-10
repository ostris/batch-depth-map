# Batch Image to Depth Map Converter

This is a simple python script to convert a batch of images from a folder to depth maps using the [MiDaS](https://github.com/isl-org/MiDaS) model.

## Installation

1. Clone this repository
2. Install the requirements using `pip install -r requirements.txt`

## Usage

```bash
python main.py <input_dir> <output_dir>
```

## Example

```bash
python main.py "/path/to/normal/images" "/path/to/depth/maps/output/folder"
```

That is it! Enjoy.

![Wonka](https://raw.githubusercontent.com/ostris/batch-depth-map/main/assets/wonka.jpg)

![Wonka Depth Map](https://raw.githubusercontent.com/ostris/batch-depth-map/main/assets/wonka_depth.jpg)

