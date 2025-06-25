# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:38:22 2025

@author: Shantanu
"""


"""Generate Synthetic employees.csv
This script creates a synthetic employees.csv dataset for the EDA learning repository, with columns EmployeeID, Name, Department, Salary, and HireDate.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
n_rows = 1000
departments = ['Sales', 'IT', 'HR', 'Marketing', 'Finance']
first_names = ['John', 'Jane', 'Alex', 'Emily', 'Michael', 'Sarah', 'David', 'Laura']
last_names = ['Smith', 'Johnson', 'Brown', 'Taylor', 'Wilson', 'Davis', 'Clark']
start_date = datetime(2015, 1, 1)
end_date = datetime(2023, 12, 31)

# Generate data
data = {
    'EmployeeID': range(1000, 1000 + n_rows),
    'Name': [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_rows)],
    'Department': [random.choice(departments) for _ in range(n_rows)],
    'Salary': np.random.uniform(30000.00, 150000.00, n_rows).round(2),
    'HireDate': [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(n_rows)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV with error handling
try:
    os.makedirs('data', exist_ok=True)
    output_path = "data/employees.csv"  # Save in data/ directory
    df.to_csv(output_path, index=False)
    absolute_path = os.path.abspath(output_path)
    print(f"Saved {n_rows} records to {absolute_path}")
except Exception as e:
    print(f"Error saving file: {e}")
