# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:32:02 2025

@author: Shantanu
"""



"""Generate Synthetic orders.csv
This script creates a synthetic orders.csv dataset for the EDA learning repository, with columns OrderID, CustomerID, ProductID, Date, Amount, and Region.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
n_rows = 1000
regions = ['North', 'South', 'East', 'West']
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Generate data
data = {
    'OrderID': range(1, n_rows + 1),
    'CustomerID': np.random.randint(1000, 2000, n_rows),
    'ProductID': np.random.randint(1, 100, n_rows),
    'Date': [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(n_rows)],
    'Amount': np.random.uniform(10, 5000, n_rows).round(2),
    'Region': [random.choice(regions) for _ in range(n_rows)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure data/ directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = "data/orders.csv"  # Save in data/ directory
df.to_csv(output_path, index=False)
print(f"Saved {n_rows} records to {output_path}")
