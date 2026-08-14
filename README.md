# Hydraulic Attention Transformer

A transformer for flood forecasting where attention strength is set by
hydraulic travel time, recomputed at every timestep, instead of by feature
similarity or a fixed mesh neighbourhood.

Status: Week 1 — data verified and inspected. No model trained yet.

## Data

FloodCastBench (Xu, Shi, Zhao & Zhu, *Scientific Data* 12, 431, 2025,
doi:10.1038/s41597-025-04725-2). Data on Zenodo,
doi:10.5281/zenodo.14017092, CC BY 4.0.

The dataset is not in this repository. Rebuild it:

    mkdir -p data && cd data
    wget -c -O FloodCastBench.zip \
      "https://zenodo.org/api/records/14017092/files/FloodCastBench.zip/content"
    md5sum FloodCastBench.zip   # c43f3009c82e212ef21a65739f4ada3d
    unzip -q FloodCastBench.zip \
      'FloodCastBench/High-fidelity flood forecasting/30m/*' \
      'FloodCastBench/Relevant data/*' -d raw/

20 GB archive, 14 GB extracted.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Scripts

- `scripts/peek_zenodo.py` — list a Zenodo record's files and sizes without downloading
- `scripts/peek_tif.py` — open one depth frame and print shape, resolution, CRS and depth range

## Documentation

See `docs/` for setup, the data card, the decisions log, and open questions.

## Known traps

- Depth frames carry no georeferencing. Cell size is 30 m and must be hard-coded.
- Filenames sort alphabetically; parse the integer and sort numerically.
- Rainfall is half-hourly against 300 s depth frames — six frames per rainfall grid.
