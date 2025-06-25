# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:35:55 2025

@author: Shantanu
"""


"""Generate Synthetic products.csv
This script creates a synthetic products.csv dataset for the EDA learning repository, with columns ProductID, ProductName, Category, Price, and Stock.
"""

import pandas as pd
import numpy as np
import random
import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
n_rows = 100
categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Toys']
product_names = [
    'Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch',
    'T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Hat',
    'Lamp', 'Chair', 'Table', 'Bedding', 'Rug',
    'Novel', 'Textbook', 'Notebook', 'Planner', 'Comics',
    'Doll', 'Puzzle', 'Board Game', 'Action Figure', 'Building Set'
]

# Generate data
data = {
    'ProductID': range(1, n_rows + 1),
    'ProductName': [random.choice(product_names) for _ in range(n_rows)],
    'Category': [random.choice(categories) for _ in range(n_rows)],
    'Price': np.random.uniform(5.00, 1000.00, n_rows).round(2),
    'Stock': np.random.randint(0, 500, n_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV with error handling
try:
    os.makedirs('data', exist_ok=True)
    output_path = "data/products.csv"  # Save in data/ directory
    df.to_csv(output_path, index=False)
    absolute_path = os.path.abspath(output_path)
    print(f"Saved {n_rows} records to {absolute_path}")
except Exception as e:
    print(f"Error saving file: {e}")
