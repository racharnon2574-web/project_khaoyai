import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


# ==============================
# FEATURE ENGINEERING
# ==============================

def create_features(ts):

    df = pd.DataFrame(ts).copy()

    # ถ้ามีหลาย column ให้ใช้เฉพาะ target y
    if df.shape[1] > 1:
        df = df[["y"]]

    df.index = pd.to_datetime(df.index)

    # Lag features
    lags = [1,2,3,4,6,8,12,26,52]

    for lag in lags:
        df[f"lag{lag}"] = df["y"].shift(lag)

    # Rolling mean
    df["roll7_mean"] = df["y"].rolling(7).mean()
    df["roll30_mean"] = df["y"].rolling(30).mean()

    # Rolling statistics
    windows = [4, 8, 12, 26]

    for w in windows:
        df[f"roll{w}_mean"] = df["y"].rolling(w).mean()
        df[f"roll{w}_std"] = df["y"].rolling(w).std()

    # Monthly seasonality
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Weekly pattern
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Weekly seasonality
    df["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Trend
    df["trend"] = range(len(df))

    df = df.dropna()

    return df


# ==============================
# TRAIN + TEST FORECAST
# ==============================

def run_xgboost(train_ts, test_ts):

    full_ts = pd.concat([train_ts, test_ts])

    df = create_features(full_ts)

    split_date = train_ts.index[-1]

    train = df[df.index <= split_date]
    test = df[df.index > split_date]

    X_train = train.drop("y", axis=1)
    y_train = train["y"]

    X_test = test.drop("y", axis=1)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    # save model for GUI
    joblib.dump(model, "xgb_model.pkl")

    forecast = model.predict(X_test)

    return forecast


# ==============================
# FORECAST FUTURE (CLI)
# ==============================

def forecast_future(last_known_ts, start_date, days, model_path="xgb_model.pkl"):

    model = joblib.load(model_path)

    history = last_known_ts.copy()
    history.index = pd.to_datetime(history.index)

    predictions = []
    forecast_dates = []

    for _ in range(days):

        df = create_features(history)

        last_row = df.iloc[-1:]
        X_last = last_row.drop("y", axis=1)

        next_pred = model.predict(X_last)[0]

        next_date = history.index[-1] + pd.Timedelta(days=1)

        predictions.append(next_pred)
        forecast_dates.append(next_date)

        history.loc[next_date] = next_pred

    return forecast_dates, predictions


# ==============================
# GUI FORECAST FUNCTION
# ==============================

def forecast_from_gui(last_known_ts, start_date, days, model_path="xgb_model.pkl"):

    model = joblib.load(model_path)

    history = last_known_ts[last_known_ts.index < pd.to_datetime(start_date)].copy()
    history.index = pd.to_datetime(history.index)

    predictions = []
    forecast_dates = []

    current_date = pd.to_datetime(start_date)

    for _ in range(days):

        df = create_features(history)

        last_row = df.iloc[-1:]
        X_last = last_row.drop("y", axis=1)

        next_pred = model.predict(X_last)[0]

        forecast_dates.append(current_date)
        predictions.append(next_pred)

        history.loc[current_date] = next_pred

        current_date += pd.Timedelta(days=1)

    return forecast_dates, predictions