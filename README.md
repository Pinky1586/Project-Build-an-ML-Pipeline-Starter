# NYC Airbnb Price Prediction Pipeline

An end-to-end, reproducible Machine Learning pipeline built with **MLflow** and **Weights & Biases (W&B)**. This project cleans raw NYC Airbnb listing data, performs automated data validation, splits the dataset, trains a Random Forest regressor with hyperparameter tuning, and evaluates the final model on test data.

---

## Project Links

* **Weights & Biases Project:** [W&B Project Dashboard](https://wandb.ai/cjernbe-western-governors-university/nyc_airbnb/overview)
* **GitHub Release Tag:** `1.0.2`

---

## Machine Learning Pipeline Architecture

The pipeline is modularized into independent MLflow execution steps:

1. **`basic_cleaning`**: Ingests raw data (`sample.csv`/`sample2.csv`), cleans price/fee columns, and filters out erroneous geospatial outliers outside NYC bounds.
2. **`data_check`**: Runs automated pytest assertions against the dataset (e.g., verifying dataset size and ensuring valid price ranges).
3. **`data_split`**: Segregates data into training/validation and test sets (`trainval_data.csv` and `test_data.csv`).
4. **`train_random_forest`**: Fits a Random Forest regression pipeline (handling imputation and One-Hot Encoding for categorical features), tracks hyperparameter experiments, and logs the serialized `model_export` artifact.
5. **`test_regression_model`**: Evaluates the production model against the holdout test dataset.

---

## How to Run

### Prerequisites
Ensure environment dependencies are active in your Conda or virtual environment:

```bash
conda activate nyc_airbnb_dev
