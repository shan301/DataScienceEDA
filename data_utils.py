# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 21:21:43 2025

@author: Shantanu
"""


"""Data Utilities
This script provides utility functions for data preprocessing in Exploratory Data Analysis (EDA). It includes functions for loading datasets, handling missing values, encoding categorical variables, and scaling numerical features, compatible with pandas, Dask, and Vaex. Designed for datasets like orders.csv, products.csv, employees.csv, sales.csv, and financials.csv.
"""

import pandas as pd
import dask.dataframe as dd
import vaex
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

"""1. Load Dataset
Load a dataset using pandas, Dask, or Vaex based on size and type."""
def load_dataset(file_path, library='pandas', parse_dates=None):
    """Load a dataset using the specified library.
    
    Args:
        file_path (str): Path to the CSV file.
        library (str): Library to use ('pandas', 'dask', 'vaex'). Default: 'pandas'.
        parse_dates (list): Columns to parse as dates.
    
    Returns:
        DataFrame: Loaded dataset (pandas.DataFrame, dask.dataframe, or vaex.DataFrame).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    if library == 'pandas':
        return pd.read_csv(file_path, parse_dates=parse_dates)
    elif library == 'dask':
        return dd.read_csv(file_path, parse_dates=parse_dates)
    elif library == 'vaex':
        return vaex.from_csv(file_path, parse_dates=parse_dates)
    else:
        raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")

# Example: Load orders.csv with pandas
orders_df = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
print(orders_df.head())  # Output: First 5 rows of orders.csv

"""2. Handle Missing Values
Fill or drop missing values in a dataset."""
def handle_missing_values(df, strategy='mean', columns=None):
    """Handle missing values in specified columns.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
        columns (list): Columns to process. If None, applies to all numeric columns.
    
    Returns:
        DataFrame: DataFrame with handled missing values.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns if isinstance(df, pd.DataFrame) else df.get_column_names(numeric=True)
    
    if isinstance(df, pd.DataFrame):
        if strategy == 'drop':
            return df.dropna(subset=columns)
        for col in columns:
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
    elif isinstance(df, dd.DataFrame):
        if strategy == 'drop':
            return df.dropna(subset=columns)
        for col in columns:
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean().compute())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median().compute())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode().compute()[0])
    elif isinstance(df, vaex.DataFrame):
        for col in columns:
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'drop':
                df = df.drop_missing(columns)
    return df

# Example: Fill missing Amounts in orders.csv
orders_clean = handle_missing_values(orders_df.copy(), strategy='mean', columns=['Amount'])
print(orders_clean['Amount'].isna().sum())  # Output: Number of missing values (should be 0)

"""3. Encode Categorical Variables
Encode categorical columns using label encoding."""
def encode_categorical(df, columns):
    """Encode categorical columns using LabelEncoder.
    
    Args:
        df: Input DataFrame (pandas only for simplicity).
        columns (list): Categorical columns to encode.
    
    Returns:
        DataFrame: DataFrame with encoded columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Encoding currently supported for pandas DataFrame only.")
    
    df_encoded = df.copy()
    for col in columns:
        df_encoded[f"{col}_Encoded"] = LabelEncoder().fit_transform(df_encoded[col])
    return df_encoded

# Example: Encode Region in products.csv
products_df = load_dataset('data/products.csv', library='pandas')
products_encoded = encode_categorical(products_df, columns=['Category'])
print(products_encoded[['Category', 'Category_Encoded']].head())  # Output: Original vs encoded Category

"""4. Scale Numerical Features
Scale numerical columns using StandardScaler."""
def scale_features(df, columns):
    """Scale numerical columns using StandardScaler.
    
    Args:
        df: Input DataFrame (pandas only for simplicity).
        columns (list): Numerical columns to scale.
    
    Returns:
        DataFrame: DataFrame with scaled columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Scaling currently supported for pandas DataFrame only.")
    
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled

# Example: Scale Price and Stock in products.csv
products_scaled = scale_features(products_df, columns=['Price', 'Stock'])
print(products_scaled[['Price', 'Stock']].head())  # Output: Scaled Price and Stock values

"""Exercises
Practice data preprocessing with the following exercises using orders.csv, products.csv, and employees.csv."""

"""Exercise 1: Load Employees Data
Load employees.csv using pandas and display the first 5 rows."""
emp_ex1 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
print(emp_ex1.head())  # Output: First 5 rows of employees.csv

"""Exercise 2: Load Orders with Dask
Load orders.csv using Dask and display the first 10 rows."""
orders_ex2 = load_dataset('data/orders.csv', library='dask', parse_dates=['Date'])
print(orders_ex2.head(10))  # Output: First 10 rows of orders.csv

"""Exercise 3: Load Products with Vaex
Load products.csv using Vaex and print column names."""
products_ex3 = load_dataset('data/products.csv', library='vaex')
print(products_ex3.get_column_names())  # Output: List of column names

"""Exercise 4: Handle Missing Salaries
Fill missing Salary values in employees.csv with the median using pandas."""
emp_ex4 = handle_missing_values(emp_ex1.copy(), strategy='median', columns=['Salary'])
print(emp_ex4['Salary'].isna().sum())  # Output: Number of missing Salary values (should be 0)

"""Exercise 5: Drop Missing Values
Drop rows with missing Amount in orders.csv using pandas."""
orders_ex5 = handle_missing_values(orders_df.copy(), strategy='drop', columns=['Amount'])
print(orders_ex5.shape)  # Output: Shape of DataFrame after dropping missing values

"""Exercise 6: Encode Department
Encode the Department column in employees.csv using label encoding."""
emp_ex6 = encode_categorical(emp_ex1, columns=['Department'])
print(emp_ex6[['Department', 'Department_Encoded']].head())  # Output: Original vs encoded Department

"""Exercise 7: Scale Salary
Scale the Salary column in employees.csv using StandardScaler."""
emp_ex7 = scale_features(emp_ex1, columns=['Salary'])
print(emp_ex7['Salary'].head())  # Output: Scaled Salary values

"""Exercise 8: Handle Missing Prices with Vaex
Fill missing Price values in products.csv with the mean using Vaex."""
products_ex8 = load_dataset('data/products.csv', library='vaex')
products_ex8 = handle_missing_values(products_ex8, strategy='mean', columns=['Price'])
print(products_ex8['Price'].isna().sum())  # Output: Number of missing Price values (should be 0)

"""Exercise 9: Load Financials with Dask
Load financials.csv with Dask and compute the number of rows."""
financials_ex9 = load_dataset('data/financials.csv', library='dask', parse_dates=['Date'])
row_count = financials_ex9.shape[0].compute()
print(f"Number of rows: {row_count}")  # Output: Total rows in financials.csv

"""Exercise 10: Encode Region in Orders
Encode the Region column in orders.csv using pandas."""
orders_ex10 = encode_categorical(orders_df, columns=['Region'])
print(orders_ex10[['Region', 'Region_Encoded']].head())  # Output: Original vs encoded Region

"""Exercise 11: Scale Amount with Pandas
Scale the Amount column in orders.csv using StandardScaler."""
orders_ex11 = scale_features(orders_df, columns=['Amount'])
print(orders_ex11['Amount'].head())  # Output: Scaled Amount values

"""Exercise 12: Handle Missing Stock with Dask
Fill missing Stock values in products.csv with the mode using Dask."""
products_ex12 = load_dataset('data/products.csv', library='dask')
products_ex12 = handle_missing_values(products_ex12, strategy='mode', columns=['Stock'])
print(products_ex12['Stock'].isna().sum().compute())  # Output: Number of missing Stock values (should be 0)

"""Exercise 13: Combine Load and Clean
Load employees.csv with pandas, fill missing Salary with mean, and encode Department."""
emp_ex13 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
emp_ex13 = handle_missing_values(emp_ex13, strategy='mean', columns=['Salary'])
emp_ex13 = encode_categorical(emp_ex13, columns=['Department'])
print(emp_ex13[['Salary', 'Department', 'Department_Encoded']].head())  # Output: Processed DataFrame

"""Exercise 14: Scale Multiple Columns
Scale Price and Stock in products.csv using pandas."""
products_ex14 = scale_features(products_df, columns=['Price', 'Stock'])
print(products_ex14[['Price', 'Stock']].head())  # Output: Scaled Price and Stock values

"""Exercise 15: Check Missing Values
Count missing values in all columns of orders.csv using pandas."""
orders_ex15 = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
missing_counts = orders_ex15.isna().sum()
print(missing_counts)  # Output: Missing value counts per column

"""Notes
- Ensure datasets (orders.csv, products.csv, employees.csv, sales.csv, financials.csv) are in the data/ directory.
- Install required libraries: `pip install pandas numpy scikit-learn dask vaex`.
- Functions support pandas, Dask, and Vaex for scalability; encoding and scaling are pandas-only for simplicity.
- Use `visualization_utils.py` for plotting and `stats_utils.py` for statistical analysis (not included here).
- For large datasets, prefer Dask or Vaex to handle memory constraints.
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass
