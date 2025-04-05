# Master's Thesis – Multi-Output Classification with Clustered Conformal Prediction

## Project Overview

This repository contains the code and experimental framework for my master's thesis on **Implementation of Multi-Output Classification with Clustered Conformal Prediction**, aiming to enhance the reliability of structured prediction tasks. The focus is on quantifying uncertainty in predictions using conformal methods, particularly in the presence of output dependencies, and evaluating their effectiveness through reproducible experiments.

The project addresses the following key areas:

- **Multi-output classification**: Tackling problems where each input is associated with multiple interdependent labels.
- **Clustered conformal prediction**: Improving prediction sets by accounting for structure within output spaces.

## Documentation

For detailed information about specific components, refer to the following documentation:

- [Data Acquisition](doc/data-acquisition.md): Downloading, organizing, and splitting the dataset for experiments.

## Getting Started

Clone the repository and run the full data pipeline:

```bash
git clone https://github.com/tmb2k01/masters-thesis.git
cd masters-thesis
bash scripts/prepare-data.sh
