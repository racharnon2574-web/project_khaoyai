from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def run_adf_test(ts):

    result = adfuller(ts)

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])


def plot_acf_pacf(ts):

    plt.figure(figsize=(10,4))
    plot_acf(ts, lags=30)
    plt.title("ACF Plot")
    plt.show()

    plt.figure(figsize=(10,4))
    plot_pacf(ts, lags=30, method="ywm")
    plt.title("PACF Plot")
    plt.show()
    