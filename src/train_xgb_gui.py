import joblib
from xgboost import XGBRegressor
from preprocessing import load_and_prepare_data
from xgb_model_gui import create_features_gui


def train_and_save_gui_model():

    ts = load_and_prepare_data("data/2021-2025.xlsx")

    df = create_features_gui(ts)

    X = df.drop("y", axis=1)
    y = df["y"]

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, "xgb_model_gui.pkl")
    print("✅ GUI Model saved as xgb_model_gui.pkl")


if __name__ == "__main__":
    train_and_save_gui_model()