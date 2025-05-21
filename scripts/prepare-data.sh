#!/bin/bash

cd "$(dirname "$0")/.."
if [ ! -d "data" ]; then
    mkdir -p data
fi

# Check if the SGVehicle dataset is already downloaded
if [ ! -f "data/SGVEHICLE_dataset.tar.gz" ]; then
    curl -L -o data/SGVEHICLE_dataset.tar.gz https://github.com/pappd90/mdc_dataset/raw/main/data/MDC_dataset.tar.gz
fi
tar -xzf data/SGVEHICLE_dataset.tar.gz -C data

# Function to download from Google Drive using file ID
download_from_gdrive() {
    FILE_ID="$1"
    DEST_NAME="$2"

    if [ ! -f "data/${DEST_NAME}" ]; then
        echo "Downloading ${DEST_NAME} from Google Drive..."
        CONFIRM=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" | \
                  grep -o 'confirm=[^&]*' | sed 's/confirm=//')

        curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o "data/${DEST_NAME}"
    else
        echo "${DEST_NAME} already exists, skipping download."
    fi
}

# Download the UTKFaces tarballs if they don't exist
download_from_gdrive "1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW" "UTKFaces_dataset_1.tar.gz"
download_from_gdrive "19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b" "UTKFaces_dataset_2.tar.gz"
download_from_gdrive "1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b" "UTKFaces_dataset_3.tar.gz"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if command -v python &>/dev/null; then
    python scripts/prepare-sgvehicle-dataset.py
    python scripts/prepare-utkfaces-dataset.py
else
    python3 scripts/prepare-sgvehicle-dataset.py
    python3 scripts/prepare-utkfaces-dataset.py
fi