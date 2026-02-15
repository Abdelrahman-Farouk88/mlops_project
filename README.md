# Water Potability Prediction — MLOps Project

An end-to-end MLOps pipeline for predicting water potability using Machine Learning, with experiment tracking, model versioning, and a web interface.

## Overview

This project predicts whether water is **safe to drink** based on 9 water quality parameters using a Random Forest classifier. The pipeline is fully automated with DVC, tracked with MLflow, and served via a Streamlit web app.

## Tech Stack

- **ML Framework**: scikit-learn (Random Forest)
- **Pipeline Orchestration**: DVC
- **Experiment Tracking**: MLflow + DagsHub
- **Model Registry**: MLflow (DagsHub-hosted)
- **Web Interface**: Streamlit
- **Version Control**: Git + DVC

## Project Structure

```
├── app.py                 <- Streamlit web interface
├── prediction.py          <- CLI prediction script
├── dvc.yaml               <- DVC pipeline definition
├── params.yaml            <- Model/data parameters
├── requirements.txt       <- Python dependencies
├── data/
│   ├── external/          <- Raw source data (water_potability.csv)
│   ├── raw/               <- Train/test split
│   └── processed/         <- Preprocessed data
├── models/                <- Trained model artifacts
├── reports/               <- Metrics and run info
│   ├── metrics.json
│   └── run_info.json
├── src/
│   ├── data/
│   │   ├── data_collection.py   <- Data ingestion & splitting
│   │   └── data_prep.py         <- Data preprocessing
│   └── model/
│       ├── model_building.py    <- Model training
│       ├── model_eval.py        <- Evaluation & MLflow logging
│       └── model_reg.py         <- Model registration
└── notebooks/             <- Experiment notebooks
```

## Pipeline Stages

The DVC pipeline runs 5 stages in sequence:

1. **data_collection** — Splits raw data into train/test sets
2. **pre_processing** — Cleans data (handles missing values)
3. **model_building** — Trains Random Forest model
4. **model_eval** — Evaluates model, logs metrics/artifacts to DagsHub MLflow
5. **model_registration** — Registers model in MLflow registry, transitions to Staging

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Abdelrahman-Farouk88/mlops_project.git
cd mlops_project

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install streamlit dvc mlflow dagshub scikit-learn pandas seaborn matplotlib pyyaml
```

## DagsHub Configuration

This project uses DagsHub for experiment tracking and DVC remote storage. To set up:

1. Create a free account at [dagshub.com](https://dagshub.com)
2. Fork this repo on DagsHub: [Abdelrahman-Farouk88/mlops_project](https://dagshub.com/Abdelrahman-Farouk88/mlops_project)
3. Get your access token from **DagsHub → Settings → Tokens**
4. Configure DVC remote authentication:

```bash
dvc remote modify myremote --local auth basic
dvc remote modify myremote --local user <YOUR_DAGSHUB_USERNAME>
dvc remote modify myremote --local password <YOUR_DAGSHUB_TOKEN>
```

5. Set DagsHub environment variables (optional, for MLflow auth):

```bash
# Windows
set MLFLOW_TRACKING_USERNAME=<YOUR_DAGSHUB_USERNAME>
set MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN>

# Linux/Mac
export MLFLOW_TRACKING_USERNAME=<YOUR_DAGSHUB_USERNAME>
export MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN>
```

## Usage

### Pull data from remote
```bash
dvc pull
```

### Run the full pipeline
```bash
dvc repro
```

### Run prediction (CLI)
```bash
python prediction.py
```

### Launch the web interface
```bash
streamlit run app.py
```

### Push data to remote
```bash
dvc push
```

## Water Quality Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| pH | Acidity/alkalinity of water | 0-14 |
| Hardness | Capacity to precipitate soap | mg/L |
| Solids | Total dissolved solids | ppm |
| Chloramines | Amount of chloramines | ppm |
| Sulfate | Amount of sulfate | mg/L |
| Conductivity | Electrical conductivity | μS/cm |
| Organic Carbon | Amount of organic carbon | ppm |
| Trihalomethanes | Amount of trihalomethanes | μg/L |
| Turbidity | Measure of light emitting property | NTU |

## Experiment Tracking

- **GitHub Repo**: [mlops_project](https://github.com/Abdelrahman-Farouk88/mlops_project)
- **MLflow UI**: [DagsHub MLflow](https://dagshub.com/Abdelrahman-Farouk88/mlops_project.mlflow)
- **DagsHub Repo**: [mlops_project](https://dagshub.com/Abdelrahman-Farouk88/mlops_project)

## License

See [LICENSE](LICENSE) for details.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
