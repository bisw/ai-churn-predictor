from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customers.csv")

output_dir = ROOT / "reports"
output_dir.mkdir(exist_ok=True)

# Chart 1: Churn distribution
df["churn"].value_counts().sort_index().plot(
    kind="bar",
    title="Customer Churn Distribution",
)
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(output_dir / "churn_distribution.png")
plt.close()

# Chart 2: Churn by contract
contract_churn = (
    df.groupby("contract_type")["churn"].mean()
)

contract_churn.plot(
    kind="bar",
    title="Churn Rate by Contract Type",
)
plt.ylabel("Churn rate")
plt.tight_layout()
plt.savefig(output_dir / "contract_churn.png")
plt.close()

# Chart 3: Tenure vs churn
df.boxplot(column="tenure_months", by="churn")
plt.suptitle("")
plt.title("Customer Tenure by Churn Status")
plt.ylabel("Tenure (months)")
plt.tight_layout()
plt.savefig(output_dir / "tenure_churn.png")
plt.close()

print("Saved 3 charts in reports/")