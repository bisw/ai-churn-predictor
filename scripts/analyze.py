from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customers.csv")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nNumerical statistics:")
print(df.describe())

print("\nChurn by contract type:")
print(
    df.groupby("contract_type")["churn"]
    .agg(["count", "mean"])
)

print("\nHigh-risk customer segment:")
high_risk = df[
    (df["support_calls"] >= 4)
    & (df["tenure_months"] < 12)
]

print(f"Customers: {len(high_risk)}")
print(f"Churn rate: {high_risk['churn'].mean():.2%}")

hmc = df[df['monthly_charges'] > 2500]
print(f"Chnages more 2k: {hmc.describe()}")