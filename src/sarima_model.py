from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_sarima(train):

    model = SARIMAX(
        train,
        order=(1,1,1),
        seasonal_order=(1,0,0,52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    return model_fit