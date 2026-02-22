import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load test data
test_df = pd.read_csv(r"data/DailyDelhiClimateTest.csv")

if "date" in test_df.columns:
    test_df["date"] = pd.to_datetime(test_df["date"])
    test_df = test_df.sort_values("date")

test_df["lag_1"] = test_df["meantemp"].shift(1)
test_df = test_df.dropna()

X_test = test_df[["lag_1"]]
y_test = test_df["meantemp"]

# Fetch the latest run ID automatically
client = mlflow.tracking.MlflowClient()
experiment_name = "Climate_Forecasting_gold"
exp = client.get_experiment_by_name(experiment_name)

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)
latest_run_id = runs[0].info.run_id
print("Using latest run ID:", latest_run_id)

# Load the model
model_uri = f"runs:/{latest_run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Predictions and KPIs
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("Test set results")
print("RMSE:", rmse)
print("MAE:", mae)