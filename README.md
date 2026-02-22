# ModelOps Assignment - managing Models under Evolving Data

## Overview

This repository contains the code and pipeline for **Assignment 4: ModelOps**, focusing on managing time-series models under evolving data conditions. In this assignment, the Gold dataset produced in Assignment 3 is used, which contains daily climate observations. 

## Setup instructions and execution flow
1. Clone the repository:
   ```bash
      git clone <your_repo_url>
      cd modelops-assignment
3. Create a Python virtual environment and install dependencies:
   ```bash
      python -m venv venv
      .\venv\Scripts\activate       # Windows
      pip install -r requirements.txt
4. Run the training script to train and log the model in MLflow:
   ```bash
      python scripts/train_model.py
5. Start the MLflow UI (optional but helps to visualize the experiments):
   ```bash
      mlflow ui
6. Run the prediction script test data:
   ```bash
      python scripts/predict_script.py

## Notes
- Training script logs models, metrics (RMSE, MAE), and dataset Git commit in MLflow.
- Prediction script uses the latest MLflow model to generate test set results.
- MLflow experiment name: Climate_Forecasting_gold.

  
