import numpy as np
from preprocessing import load_and_prepare_data
from xgb_model import run_xgboost
from evaluation import evaluate_forecast


def show_user_forecast(forecast_value, smape):

    accuracy = 100 - smape

    lower = forecast_value * (1 - smape/100)
    upper = forecast_value * (1 + smape/100)

    print("\n" + "="*60)
    print("📊 Tourism Forecast Result")
    print("="*60)

    print(f"👥 Expected Tourists: {int(forecast_value):,} people")

    print(f"\n🎯 Model Accuracy: {accuracy:.2f}%")
    print(f"⚠️ Expected Error Range: ±{smape:.2f}%")

    print("\n📈 Estimated Range:")
    print(f"   Minimum: {int(lower):,} people")
    print(f"   Maximum: {int(upper):,} people")

    print("\n💡 Interpretation:")
    print(f"If prediction error occurs, the number of tourists")
    print(f"may vary by approximately {int(abs(forecast_value - lower)):,} people.")

    print("="*60)


def main():

    ts = load_and_prepare_data("data/2021-2025.xlsx")

    # 🔮 ใช้ข้อมูลทั้งหมด train แล้ว forecast สัปดาห์ถัดไป
    train = ts

    # forecast 1 step ahead
    forecast = run_xgboost(train, train.tail(1))

    forecast_value = np.expm1(forecast[-1])

    # ใช้ sMAPE จากบทที่ 4
    best_smape = 14.41

    show_user_forecast(forecast_value, best_smape)


if __name__ == "__main__":
    main()