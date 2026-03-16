import pandas as pd
import numpy as np
import holidays


def load_and_prepare_data(filepath):

    df = pd.read_excel(filepath, sheet_name="tourist")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    # =============================
    # Target (log scale)
    # =============================
    df["y"] = np.log1p(df["TOTAL"])

    # =============================
    # Holiday (วันหยุดราชการไทย)
    # =============================
    th_holidays = holidays.Thailand()

    df["is_holiday"] = df.index.to_series().apply(
        lambda x: 1 if x in th_holidays else 0
    )

    # =============================
    #  New Year (แบบมีระดับความแรง)
    # =============================
    df["newyear_strength"] = 0.0

    for date in df.index:

        if date.month == 12 and date.day == 30:
            df.loc[date, "newyear_strength"] = 1.0

        elif date.month == 12 and date.day == 31:
            df.loc[date, "newyear_strength"] = 0.9

        elif date.month == 1 and date.day == 1:
            df.loc[date, "newyear_strength"] = 0.8

        elif date.month == 1 and date.day == 2:
            df.loc[date, "newyear_strength"] = 0.6

        elif date.month == 1 and date.day == 3:
            df.loc[date, "newyear_strength"] = 0.4

    # =============================
    #  Songkran
    # =============================
    df["is_songkran"] = df.index.to_series().apply(
        lambda x: 1 if x.month == 4 and x.day in [13, 14, 15] else 0
    )


    # =============================
    # Before / After Holiday Effect
    # =============================
    df["before_holiday"] = df["is_holiday"].shift(-1).fillna(0)
    df["after_holiday"] = df["is_holiday"].shift(1).fillna(0)

    # =============================
    # Long weekend
    # =============================
    df["long_weekend"] = (
        (df["is_holiday"] == 1) &
        (df.index.dayofweek >= 4)
    ).astype(int)

    # =============================
    # Return เฉพาะ feature ที่ใช้
    # =============================
    ts_daily = df[
        [
            "y",
            "is_holiday",
            "newyear_strength",
            "is_songkran",
            "before_holiday",
            "after_holiday",
            "long_weekend"
        ]
    ]

    return ts_daily