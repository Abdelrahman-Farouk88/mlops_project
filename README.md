# Water Potability Prediction — MLOps Project



## Live Deployment


- **Web Interface:** [Water Potability Prediction App](https://mlopsproject-production-132d.up.railway.app/)
- **API:** [Water Potability Prediction API](https://mlopsproject-production-067f.up.railway.app/)

---

## Project Summary

This project is an end-to-end MLOps pipeline for predicting water potability using a Random Forest classifier. It includes:
- Automated data processing and model training with DVC
- Experiment tracking and model registry with MLflow/DagsHub
- Real-time predictions via FastAPI
- User-friendly web interface via Streamlit
- CLI and scripts for model management
- Unit tests for model loading and performance

## Key Features

- **End-to-end MLOps pipeline**: Data ingestion, preprocessing, model training, evaluation, registration, and promotion.
- **API Backend**: FastAPI service for real-time predictions ([API/main.py](API/main.py)).
- **Web Interface**: Streamlit frontend ([Interface/app.py](Interface/app.py)) for user-friendly predictions.
- **CLI Prediction**: Command-line script ([prediction.py](prediction.py)) for quick model inference.
- **Model Promotion**: Script ([scripts/production.py](scripts/production.py)) to automate model stage transitions.
- **Testing**: Unit tests ([tests/model_test.py](tests/model_test.py)) for model loading and performance.

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
├── API/
│   └── main.py            <- FastAPI backend for predictions
├── Interface/
│   └── app.py             <- Streamlit web interface
├── prediction.py          <- CLI prediction script
├── scripts/
│   └── production.py      <- Model promotion script
├── tests/
│   └── model_test.py      <- Model unit tests
├── dvc.yaml               <- DVC pipeline definition
├── params.yaml            <- Model/data parameters
├── requirements.txt       <- Pipeline dependencies (core)
├── requirements-dev.txt   <- Dev/docs/web dependencies
├── data/
│   ├── external/          <- Raw source data (water_potability.csv)
│   ├── raw/               <- Train/test split
│   └── processed/         <- Preprocessed data
├── models/                <- Trained model artifacts
├── reports/
│   ├── figures/           <- Confusion matrix plots
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

# Install core pipeline dependencies
pip install -r requirements.txt

# (Optional) For development, docs, or web interface:
pip install -r requirements-dev.txt
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


#
## Web Interface Usage

The Streamlit app is deployed at:
> https://mlopsproject-production-132d.up.railway.app/

Enter the 9 water quality parameters and click **Predict Potability** to get instant results on whether the water is safe to drink.

---

## Usage

### Pull data from remote
```bash
dvc pull
```

### Run the full pipeline
```bash
dvc repro
```

### Launch the web interface
```bash
streamlit run Interface/app.py
```

### Run the FastAPI backend
```bash
uvicorn API.main:app --reload
```


### Promote model to Production stage
```bash
python scripts/production.py
```

### Run tests
```bash
python -m unittest tests/model_test.py
```

#
## API Usage (Live)

The FastAPI backend is deployed at:
> https://mlopsproject-production-067f.up.railway.app/

### Endpoints

- `GET /` — Health message
- `POST /predict` — Predict potability

#### Example: Predict Potability

Send a POST request to `/predict` with JSON body:

```json
{
  "ph": 7.0,
  "Hardness": 200.0,
  "Solids": 20000.0,
  "Chloramines": 7.0,
  "Sulfate": 330.0,
  "Conductivity": 420.0,
  "Organic_carbon": 14.0,
  "Trihalomethanes": 66.0,
  "Turbidity": 4.0
}
```

Response:

```json
{
  "potable": 1
}
```


## Docker & Deployment

This project supports containerized deployment using Docker and Docker Compose:

- **Dockerfile**: Builds the FastAPI backend container.
- **Dockerfile.ui**: Builds the Streamlit web interface container.
- **docker-compose.yml**: Orchestrates both containers for local or cloud deployment.

To build and run locally:

```bash
docker-compose up --build
```

---

### API Usage Example

**POST /predict**

Send a JSON payload with water parameters to get a prediction:

```json
{
  "ph": 7.0,
  "Hardness": 200.0,
  "Solids": 20000.0,
  "Chloramines": 7.0,
  "Sulfate": 330.0,
  "Conductivity": 420.0,
  "Organic_carbon": 14.0,
  "Trihalomethanes": 66.0,
  "Turbidity": 4.0
}
```

**Response:**

```json
{
  "prediction": 1,
  "result": "Water is Potable (Safe to drink)"
}
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

## Continuous Integration (CI)

This project uses GitHub Actions for CI. On every push, the pipeline:
- Installs dependencies
- Pulls DVC-tracked data from remote
- Ensures output directories exist
- Runs the DVC pipeline

**Troubleshooting DVC in CI:**
If you see errors like `failed to pull data from the cloud` or `missing cache files`, make sure you have run `dvc push` locally to upload all tracked files to your DVC remote. CI can only pull files that exist in the remote storage.

**Secrets:**
- Add your DagsHub username and token as GitHub repository secrets:
  - `DAGSHUB_USERNAME`
  - `DAGSHUB_TOKEN`

See `.github/workflows/ci.yaml` for details.

## License

See [LICENSE](LICENSE) for details.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
