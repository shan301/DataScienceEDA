import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path

# Initialize Faker and set seeds
faker = Faker()
Faker.seed(42)
np.random.seed(42)

# Generate 1000 records
n_records = 1000
data = {
    'ID': range(1, n_records + 1),
    'Name': [faker.name() for _ in range(n_records)],
    'Age': np.random.normal(35, 10, n_records).astype(int).clip(18, 80),
    'Salary': (np.random.lognormal(10.5, 0.5, n_records) + np.random.normal(35, 10, n_records) * 100).round(2),
    'City': np.random.choice(['New York', 'Chicago', 'Los Angeles', 'Houston', 'Miami'], n_records, p=[0.3, 0.25, 0.2, 0.15, 0.1])
}
df = pd.DataFrame(data)

# Save to CSV
output_path = "customers.csv"  # Save in current directory
df.to_csv(output_path, index=False)
print(f"Saved {n_records} records to {output_path}")