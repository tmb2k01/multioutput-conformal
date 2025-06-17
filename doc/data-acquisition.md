# Data Acquisition

This project uses two datasets:

1. **SGVehicle Dataset** (formerly MDC): A synthetic vehicle dataset hosted on GitHub ([link](https://github.com/pappd90/mdc_dataset/raw/main/data/MDC_dataset.tar.gz)), containing images with labels embedded in the filenames.
2. **UTKFace Dataset**: A large-scale face dataset with age, gender, and race annotations, downloaded from Google Drive in three parts.

The data acquisition process is automated via shell scripts and Python scripts. The main script (`scripts/prepare-data.sh`) handles both datasets.

## SGVehicle Dataset

The SGVehicle dataset acquisition process:

- Creates a `data/SGVehicle/` directory if it does not exist
- Downloads the `MDC_dataset.tar.gz` archive from GitHub if not already present
- Extracts the archive contents into the `data/SGVehicle/` directory
- Splits the data into training (60%), validation (10%), testing (15%), and calibration (15%) sets
- Uses the filename structure (e.g., `color_type_id.png`) to extract label information
- For each split, copies image files into corresponding `images/` folders and writes matching label files into `labels/` folders, each containing two space-separated label values (color and type)

## UTKFace Dataset

The UTKFace dataset acquisition process:

- Creates a `data/UTKFace/` directory if it does not exist
- Downloads three parts of the dataset from Google Drive using `gdown`
- Extracts each part into the `data/UTKFace/` directory
- Merges the parts into a single dataset
- Splits the data into training (60%), validation (10%), testing (15%), and calibration (15%) sets
- Uses the filename structure (e.g., `age_gender_race_id.jpg`) to extract label information
- For each split, copies image files into corresponding `images/` folders and writes matching label files into `labels/` folders, each containing two space-separated label values (gender and race)

The resulting directory structure for each dataset looks like this:

```plaintext
data/
├── SGVehicle/
│   ├── MDC_dataset.tar.gz
│   ├── Gen_img/
│   │   └── *.png
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── calib/
│       ├── images/
│       └── labels/
└── UTKFace/
    ├── UTKFace_dataset_*.tar.gz
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
bash scripts/prepare-data.sh
```

This will handle the full data download, extraction, and preparation process for both datasets.
