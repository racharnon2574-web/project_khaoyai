import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


# เอาข้อมูลที่ได้จาก excel มาแปลง
def create_features(ts):

    df = pd.DataFrame(ts).copy()

    # ถ้ามีหลาย column ให้ใช้เฉพาะ target y
    if df.shape[1] > 1:
        df = df[["y"]]

    df.index = pd.to_datetime(df.index)

    # Lag features (ค่าของอดีต)
    # ใช้เพื่อให้โมเดลเรียนรู้ temporal dependency
    lags = [1,2,3,4,6,8,12,26,52]

    # lag_k = y(t-k)
    for lag in lags:
        df[f"lag{lag}"] = df["y"].shift(lag)

    # Rolling mean
    # ใช้เพื่อให้โมเดลเรียนรู้ temporal dependency
    df["roll7_mean"] = df["y"].rolling(7).mean() #ค่าเฉลี่ยย้อนหลัง 7 วัน
    df["roll30_mean"] = df["y"].rolling(30).mean()

    # Rolling statistics
    # การคำนวณจากข้อมูลย้อนหลังเป็นช่วง ๆ
    windows = [4, 8, 12, 26]

    for w in windows:
        df[f"roll{w}_mean"] = df["y"].rolling(w).mean()
        df[f"roll{w}_std"] = df["y"].rolling(w).std() # std  = ความผันผวนของข้อมูล

    # Monthly seasonality
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Weekly pattern
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Weekly seasonality
    # แปลง day_of_week เป็น cyclic feature
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

    split_date = train_ts.index[-1] #วันสุดท้ายของ train

    train = df[df.index <= split_date] #แยก train
    test = df[df.index > split_date] #แยก test

    X_train = train.drop("y", axis=1) # X คือ features
    y_train = train["y"] # y คือ target

    X_test = test.drop("y", axis=1)

    model = XGBRegressor(
        n_estimators=500, # n_estimators = จำนวนต้นไม้
        learning_rate=0.05, # learning_rate = ขนาดการเรียนรู้
        max_depth=4, # max_depth = ความลึกของต้นไม้
        subsample = 0.8,
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

    model = joblib.load(model_path) #โหลดโมเดล

    history = last_known_ts.copy() #ใช้ข้อมูลล่าสุด
    history.index = pd.to_datetime(history.index)

    predictions = []
    forecast_dates = []

    #forecast ทีละวัน
    for _ in range(days):

        df = create_features(history)

        last_row = df.iloc[-1:]
        X_last = last_row.drop("y", axis=1)

        next_pred = model.predict(X_last)[0]

        next_date = history.index[-1] + pd.Timedelta(days=1) #เพิ่มวันที่ใหม่

        predictions.append(next_pred)
        forecast_dates.append(next_date)

        history.loc[next_date] = next_pred #เพิ่มค่า forecast เพื่อให้ ใช้ forecast เป็น input รอบต่อไป

    return forecast_dates, predictions


# ==============================
# GUI FORECAST FUNCTION
# ==============================

def forecast_from_gui(last_known_ts, start_date, days, model_path="xgb_model.pkl"):

    model = joblib.load(model_path)

    #เลือกข้อมูลก่อนวัน forecast
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