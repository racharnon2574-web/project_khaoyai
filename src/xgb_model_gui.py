import pandas as pd
import numpy as np
import joblib


# =============================
# Feature สำหรับ GUI
# =============================
def create_features_gui(ts):

    df = ts.copy()

    df.index = pd.to_datetime(df.index)

    # lag
    lags = [1,2,3,4,6,8,12,26,52]
    for lag in lags:
        df[f"lag{lag}"] = df["y"].shift(lag)

    # rolling
    windows = [4,8,12,26]
    for w in windows:
        df[f"roll{w}_mean"] = df["y"].rolling(w).mean()
        df[f"roll{w}_std"] = df["y"].rolling(w).std()

    # month
    df["month_sin"] = np.sin(2*np.pi*df.index.month/12)
    df["month_cos"] = np.cos(2*np.pi*df.index.month/12)

    # 🔥 เพิ่ม feature รายวัน
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["day_sin"] = np.sin(2*np.pi*df.index.dayofweek/7)
    df["day_cos"] = np.cos(2*np.pi*df.index.dayofweek/7)

    df = df.dropna()

    return df

# =============================
# Direct Forecast สำหรับ GUI
# =============================
def forecast_from_gui(ts, start_date, days, model_path="xgb_model_gui.pkl"):

    model = joblib.load(model_path)

    ts = ts.copy()
    ts.index = pd.to_datetime(ts.index)

    history = ts[ts.index < pd.to_datetime(start_date)].copy()

    forecast_dates = []
    predictions = []

    current_date = pd.to_datetime(start_date)

    for _ in range(days):

        df = create_features_gui(history)

        if len(df) == 0:
            break

        last_row = df.iloc[-1:]
        X_last = last_row.drop("y", axis=1)

        next_pred = model.predict(X_last)[0]

        forecast_dates.append(current_date)
        predictions.append(next_pred)

        # ✅ feed prediction กลับเข้า history
        history.loc[current_date] = next_pred

        current_date += pd.Timedelta(days=1)

    return forecast_dates, predictions