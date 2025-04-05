# Data Acquisition

This project uses a dataset hosted on GitHub ([link](https://github.com/pappd90/mdc_dataset/raw/main/data/MDC_dataset.tar.gz)), containing synthetic images with labels embedded in the filenames. The data acquisition process is automated via a shell script (`scripts/acquire-data.sh`) and a Python script (`scripts/prepare-dataset.py`).

The acquisition script performs the following operations:

- Creates a `data/` directory if it does not exist.
- Downloads the `MDC_dataset.tar.gz` archive from a remote repository if it's not already present.
- Extracts the archive contents into the `data/` directory, resulting in a folder `data/Gen_img/` containing all image files.

Once the archive is extracted, the Python script splits the data into training (60%), validation (10%), testing (15%), and calibration (15%) sets. It uses the filename structure (e.g., `color_type_id.png`) to extract label information. For each split, the script copies the image files into corresponding `images/` folders and writes matching label files into `labels/` folders, each containing two space-separated label values (color and type).

The resulting directory structure looks like this:

```
data/
├── MDC_dataset.tar.gz
├── Gen_img/
│   └── *.png
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── calib/
    ├── images/
    └── labels/
```

To run the full pipeline from the project root, simply execute:

```bash
bash scripts/acquire-data.sh
```

This will handle the full data download, extraction, and preparation process.
