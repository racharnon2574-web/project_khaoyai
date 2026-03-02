import pandas as pd
from prophet import Prophet


def run_prophet(train_ts, test_ts):

    train_df = pd.DataFrame({
        "ds": train_ts.index,
        "y": train_ts.values
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train_df)

    future = pd.DataFrame({
        "ds": test_ts.index
    })

    forecast = model.predict(future)

    return forecast["yhat"].values