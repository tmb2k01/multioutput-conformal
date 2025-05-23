#!/bin/bash

cd "$(dirname "$0")/.."
if [ ! -d "data" ]; then
    mkdir -p data
fi

if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if the SGVehicle dataset is already downloaded
if [ ! -f "data/SGVEHICLE_dataset.tar.gz" ]; then
    curl -L -o data/SGVEHICLE_dataset.tar.gz https://github.com/pappd90/mdc_dataset/raw/main/data/MDC_dataset.tar.gz
else
    echo "SGVEHICLE_dataset.tar.gz already exists, skipping download."
fi
if [ ! -d "data/SGVehicle" ]; then
    mkdir -p data/SGVehicle
fi
tar -xzf data/SGVEHICLE_dataset.tar.gz -C data/SGVehicle

if command -v python &>/dev/null; then
    python scripts/download_utkfaces.py
else
    python3 scripts/download_utkfaces.py
fi
if [ ! -d "data/UTKFaces" ]; then
    mkdir -p data/UTKFaces
fi
tar -xzf data/UTKFaces_dataset_1.tar.gz -C data/UTKFaces
tar -xzf data/UTKFaces_dataset_2.tar.gz -C data/UTKFaces
tar -xzf data/UTKFaces_dataset_3.tar.gz -C data/UTKFaces



if command -v python &>/dev/null; then
    python scripts/prepare-sgvehicle-dataset.py
    python scripts/prepare-utkfaces-dataset.py
else
    python3 scripts/prepare-sgvehicle-dataset.py
    python3 scripts/prepare-utkfaces-dataset.py
fi