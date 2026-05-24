# IGSTGNN: Incident-Guided Spatiotemporal Traffic Forecasting

This repository contains the official implementation of **IGSTGNN (Incident-Guided Spatiotemporal Graph Neural Network)** for **incident-aware traffic forecasting**. IGSTGNN explicitly models how newly occurring incidents disrupt traffic dynamics, including **heterogeneous spatial influence** and **temporal impact decay**.

## Overview

Most spatiotemporal forecasting models learn patterns purely from historical traffic time series, but **non-recurrent incidents** (e.g., crashes, hazards, adverse weather) can cause abrupt distribution shifts that are hard to infer from history alone. IGSTGNN addresses this by injecting **incident context** into the traffic representation and explicitly modeling **how the incident influence dissipates over the prediction horizon**.

## Framework

![IGSTGNN Framework](img/framework.png)

IGSTGNN consists of three stages:

- **Incident-Context Spatial Fusion (ICSF)**: fuses incident attributes, sensor meta-features, and the latest traffic state under a pre-defined incident–sensor spatial relationship tensor to produce incident-aware node representations.
- **Spatio-temporal (ST) modeling backbone**: captures incident-conditioned spatiotemporal dependency and propagation in the traffic graph.
- **Temporal Incident Impact Decay (TIID)**: models the long-term decaying effect of incident impact and refines multi-step forecasts.

## Results

![Performance Comparison](img/performance.png)

## Dataset

You can download the prepared dataset used in this repository from Kaggle:

- **Kaggle**: [IncidentWithTraffic4Alameda](https://www.kaggle.com/datasets/lixiangfan/incidentwithtraffic4alameda)

After downloading, place the files under `IGSTGNN/data/<dataset_name>/` in the following format:

```
data/
└── <dataset_name>/
    ├── combined_distance_matrix.npy
    ├── combined_distance_norm_params.json
    ├── desc_mapping.json
    ├── incidents_data.npy
    ├── stats.npz
    └── type_mapping.json
```

## Project Structure

```
IGSTGNN/
├── data/                          # Dataset directory
│   └── xtraffic/
├── experiments/                   # Experiment configurations and entry points
│   └── IGSTGNN/
│       ├── main.py
│       └── run.sh
├── img/                           # Figures and visualizations
│   ├── framework.png
│   ├── performance.png
│   └── Incidents_heatmap/
├── src/                           # Source code
│   ├── base/                      # Base classes
│   ├── engines/                   # Training engines
│   ├── models/                    # Model implementations
│   └── utils/                     # Utility functions
├── LICENSE                        # MIT License
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## Quick Start

### Environment

Install dependencies (recommended: Python 3.8+):

```bash
pip install -r requirements.txt
```

### Training

Train IGSTGNN with incident information enabled:

```bash
python experiments/IGSTGNN/main.py \
  --device cuda:0 \
  --dataset "Alameda" \
  --model_name igstgnn \
  --seed 2023 \
  --bs 48 \
  --incident \
  --use_sensor_info
```

### Running Scripts

Run multiple experiment presets:

```bash
bash experiments/IGSTGNN/run.sh
```

## Input Features

The model uses **historical traffic time series** plus two heterogeneous contextual sources:

- **Sensor meta-features**: roadway/sensor attributes (e.g., road type, speed limit, latitude/longitude, freeway name, postmile).
- **Incident information**: incident intrinsic/contextual attributes (e.g., type, description, relative position, holiday) and spatial anchoring (e.g., latitude/longitude, freeway name, postmile).

## Visualization

Incident distribution heatmaps:

![Alameda County Incidents](img/Incidents_heatmap/Alameda.png)

![Contra Costa County Incidents](img/Incidents_heatmap/ContraCosta.png)

![Orange County Incidents](img/Incidents_heatmap/Orange.png)

## License

This project is licensed under the MIT License (see `LICENSE`).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{fan2026igstgnn,
  title = {Incident-Guided Spatiotemporal Traffic Forecasting},
  author = {Fan, Lixiang and Li, Bohao and Zou, Tao and Du, Bowen and Ye, Junchen},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD '26)},
  year = {2026},
  doi = {10.1145/3770854.3780215},
  isbn = {979-8-4007-2258-5/2026/08},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3770854.3780215}
}
```