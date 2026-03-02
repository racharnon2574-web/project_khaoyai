import pandas as pd

df = pd.read_excel("data/2021-2025.xlsx", sheet_name="tourist")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

print(df.loc["2024-12-25":"2025-01-07"])