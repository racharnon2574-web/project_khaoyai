import numpy as np
from datetime import datetime


def show_user_forecast(forecast_value, smape):

    accuracy = 100 - smape

    lower = forecast_value * (1 - smape/100)
    upper = forecast_value * (1 + smape/100)

    print("\n" + "="*60)
    print("📊 Tourism Forecast Result")
    print("="*60)

    print(f"📅 Forecast Date: {datetime.today().date()}")
    print(f"👥 Expected Tourists: {int(forecast_value):,} people")

    print(f"\n🎯 Model Accuracy: {accuracy:.2f}%")
    print(f"⚠️ Expected Error Range: ±{smape:.2f}%")

    print("\n📈 Estimated Range:")
    print(f"   Minimum: {int(lower):,} people")
    print(f"   Maximum: {int(upper):,} people")

    print("\n💡 Interpretation:")
    print(f"If prediction error occurs, the number of tourists")
    print(f"may increase or decrease by approximately")
    print(f"{int(abs(forecast_value - lower)):,} people.")

    print("="*60)