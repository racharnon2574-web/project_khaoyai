import numpy as np

def evaluate_forecast(actual, forecast):

    actual = np.array(actual)
    forecast = np.array(forecast)

    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))

    smape = np.mean(
        2 * np.abs(forecast - actual)
        / (np.abs(actual) + np.abs(forecast))
    ) * 100

    return mae, rmse, smape