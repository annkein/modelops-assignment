import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import subprocess

# Load gold dataset
df = pd.read_csv("data/climate_gold.csv")

# Convert date column to datetime if exists
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# Create simple lag feature
df["lag_1"] = df["meantemp"].shift(1)
df = df.dropna()

X = df[["lag_1"]]
y = df["meantemp"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

# Set MLflow experiment
mlflow.set_experiment("Climate_Forecasting_gold")

# Fetch the current data commit hash
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

# Check the last MLflow run
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("Climate_Forecasting_gold")

if exp is None:
    # If doesn't exist, create new
    experiment_id = client.create_experiment("Climate_Forecasting_gold")
else:
    experiment_id = exp.experiment_id

runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

last_data_commit = None
if runs:
    last_run = runs[0]
    last_data_commit = last_run.data.params.get("data_commit")

# If the commit is the same, stop
if last_data_commit == commit_hash:
    print(f"No new data. Last run already used commit {commit_hash}")
    exit(0)

with mlflow.start_run():
    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Calculate KPIs
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    # Log the parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("data_commit", commit_hash)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    print("Model training complete!")
    print("RMSE:", rmse)
    print("MAE:", mae)