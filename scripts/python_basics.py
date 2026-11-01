from dataclasses import dataclass


@dataclass
class Customer:
    customer_id: int
    monthly_charges: float
    tenure_months: int


def annual_revenue(customer: Customer) -> float:
    return customer.monthly_charges * 12


customer = Customer(
    customer_id=101,
    monthly_charges=1500.0,
    tenure_months=18,
)

print(f"Customer: {customer.customer_id}")
print(f"Annual revenue: {annual_revenue(customer):.2f}")