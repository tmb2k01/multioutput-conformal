#!/bin/bash

cd "$(dirname "$0")/.."
if [ ! -d "data" ]; then
    mkdir -p data
fi
if [ ! -f "data/MDC_dataset.tar.gz" ]; then
    curl -L -o data/MDC_dataset.tar.gz https://github.com/pappd90/mdc_dataset/raw/main/data/MDC_dataset.tar.gz
fi
tar -xzf data/MDC_dataset.tar.gz -C data

python3 scripts/prepare-dataset.py