import joblib
import pandas as pd
from preprocessing import load_and_prepare_data
from xgb_model import create_features
from xgboost import XGBRegressor

def train_and_save_model():
    ts = load_and_prepare_data("data/2021-2025.xlsx")
    df = create_features(ts)
    X = df.drop("y", axis=1)
    y = df["y"]

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        random_state=42
    )

    model.fit(X, y)
    joblib.dump(model, "xgb_model.pkl")
    print("✅ Model saved as xgb_model.pkl")


if __name__ == "__main__":
    train_and_save_model()