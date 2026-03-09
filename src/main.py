import pandas as pd
import numpy as np

from preprocessing import load_and_prepare_data
from sarima_model import run_sarima
from prophet_model import run_prophet
from xgb_model import run_xgboost
from lstm_model import run_lstm
from evaluation import evaluate_forecast


def check_forecast_by_date(date, dates, actual, forecast):

    date = pd.to_datetime(date)

    if date not in dates:
        print("Date not found in test data")
        return

    idx = list(dates).index(date)

    a = actual[idx]
    f = forecast[idx]

    error = a - f
    error_pct = (error / a) * 100

    print(f"\n===== {date.date()} =====")
    print(f"Actual   : {int(a)}")
    print(f"Forecast : {int(f)}")
    print(f"Error    : {int(error)}")
    print(f"Error %  : {error_pct:.2f}%")

def main():

    ts = load_and_prepare_data("data/2021-2025.xlsx")

    train = ts.loc["2021":"2024"]
    test = ts.loc["2025"]

    target_train = train["y"]
    target_test = test["y"]

    print("Train length:", len(train))
    print("Test length :", len(test))

    # แปลง test กลับ scale จริง
    test_actual = np.expm1(target_test)

    results = []

    # SARIMA
    sarima_model = run_sarima(train["y"])
    sarima_forecast = sarima_model.forecast(steps=len(test))
    sarima_forecast = np.expm1(sarima_forecast)

    results.append(["SARIMA", *evaluate_forecast(test_actual, sarima_forecast)])

    # Prophet
    prophet_forecast = run_prophet(target_train, target_test)
    prophet_forecast = np.expm1(prophet_forecast)

    results.append(["Prophet", *evaluate_forecast(test_actual, prophet_forecast)])

    # XGBoost
    xgb_forecast = run_xgboost(target_train, target_test)
    xgb_forecast = np.expm1(xgb_forecast)

    results.append(["XGBoost", *evaluate_forecast(test_actual, xgb_forecast)])

    # LSTM
    lstm_forecast = run_lstm(target_train, target_test)
    lstm_forecast = np.expm1(lstm_forecast)

    results.append(["LSTM", *evaluate_forecast(test_actual, lstm_forecast)])

    # ตารางเปรียบเทียบ
    results_df = pd.DataFrame(
        results,
        columns=["Model", "MAE", "RMSE", "sMAPE"]
    )

    # format ตัวเลข
    results_df["MAE"] = results_df["MAE"].round(2)
    results_df["RMSE"] = results_df["RMSE"].round(2)
    results_df["sMAPE"] = results_df["sMAPE"].round(2).astype(str) + "%"

    # เรียงจากดีที่สุด
    results_df = results_df.sort_values(by="RMSE").reset_index(drop=True)

    # เพิ่มอันดับ 1 2 3 4
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    print("\nModel Comparison")
    print("=" * 50)
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]["Model"]

    if best_model == "XGBoost":
        best_forecast = xgb_forecast

    elif best_model == "SARIMA":
        best_forecast = sarima_forecast

    elif best_model == "Prophet":
        best_forecast = prophet_forecast

    elif best_model == "LSTM":
        best_forecast = lstm_forecast

    check_date = "2025-01-1"

    check_forecast_by_date(
        check_date,
        target_test.index,
        test_actual,
        best_forecast
    )

if __name__ == "__main__":
    main()