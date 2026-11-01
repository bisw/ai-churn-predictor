from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customers.csv"


def generate_customers(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    tenure = rng.integers(1, 73, size=n)
    monthly_charges = rng.uniform(300, 3000, size=n)
    support_calls = rng.poisson(2, size=n)
    contract_type = rng.choice(
        ["monthly", "annual"],
        size=n,
        p=[0.7, 0.3],
    )

    # Synthetic probability-generating process.
    logits = (
        -1.2
        - 0.035 * tenure
        + 0.0006 * monthly_charges
        + 0.35 * support_calls
        + 0.9 * (contract_type == "monthly")
    )

    probability = 1 / (1 + np.exp(-logits))
    churn = rng.binomial(1, probability)

    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "tenure_months": tenure,
        "monthly_charges": monthly_charges.round(2),
        "support_calls": support_calls,
        "contract_type": contract_type,
        "churn": churn,
    })


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_customers()
    df.to_csv(DATA_PATH, index=False)

    print(f"Generated {len(df)} customers")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"Saved to {DATA_PATH}")