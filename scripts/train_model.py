import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

mlflow.set_experiment("Climate_Forecasting_gold")

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)

    mlflow.sklearn.log_model(model, "model")

    print("RMSE:", rmse)
    print("MAE:", mae)