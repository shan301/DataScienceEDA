# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 12:53:22 2025

@author: Shantanu
"""

"""Data Loading and Overview
This module provides utility functions for loading datasets and generating initial overviews, essential for exploratory data analysis (EDA) in data science workflows.

1. Data Loading
Functions to load data from common file formats such as CSV, Excel, JSON, and Parquet."""
from pathlib import Path
import pandas as pd
import os

def load_data(file_path):
    """
    Load data from various file formats.
    Args:
        file_path (str): Path to the data file
    Returns:
        pandas.DataFrame: Loaded dataset or None if loading fails
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        print(f"Successfully loaded {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

load_data("../data/customers.csv")  # Output: Successfully loaded ../data/customers.csv

"""2. Dataset Shape and Columns
Functions to retrieve and display the shape and column names of a dataset."""
def get_shape_and_columns(df):
    """
    Print the shape and column names of a dataset.
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is None:
        print("No data to analyze")
        return
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

# Example: get_shape_and_columns(load_data("../data/customers.csv"))
# Output: Shape: 1000 rows, 5 columns
# Columns: ['ID', 'Name', 'Age', 'Salary', 'City']

"""3. Data Types Inspection
Functions to inspect and summarize data types of dataset columns."""
def inspect_data_types(df):
    """
    Summarize the data types of each column.
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is None:
        print("No data to analyze")
        return
    print("Data Types:")
    print(df.dtypes)

# Example: inspect_data_types(load_data("../data/customers.csv"))
# Output: Data Types:
# ID        int64
# Name     object
# Age       int64
# Salary   float64
# City     object
# dtype: object

"""4. Initial Data Preview
Functions to preview the first few rows of a dataset."""
def preview_data(df, n=5):
    """
    Display the first n rows of a dataset.
    Args:
        df (pandas.DataFrame): Input dataset
        n (int): Number of rows to display
    """
    if df is None:
        print("No data to analyze")
        return
    print(f"First {n} Rows:")
    print(df.head(n))

# Example: preview_data(load_data("../data/customers.csv"), 3)
# Output: First 3 Rows:
# (Shows first 3 rows of customers.csv)

"""5. Missing Values Summary
Functions to summarize missing values in a dataset."""
def summarize_missing_values(df):
    """
    Report columns with missing values and their counts.
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is None:
        print("No data to analyze")
        return
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found")
    else:
        print("Missing Values:")
        print(missing)

# Example: summarize_missing_values(load_data("../data/customers.csv"))
# Output: No missing values found

"""6. Basic Statistical Summary
Functions to provide statistical summaries for numerical columns."""
def basic_statistical_summary(df):
    """
    Generate statistical summary for numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is None:
        print("No data to analyze")
        return
    print("Statistical Summary:")
    print(df.describe())

# Example: basic_statistical_summary(load_data("../data/customers.csv"))
# Output: Statistical Summary:
# (Shows mean, std, min, max, etc. for numerical columns like Age, Salary)

"""7. Unique Values Count
Functions to count unique values per column."""
def count_unique_values(df):
    """
    Count unique values per column in the dataset.
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is None:
        print("No data to analyze")
        return
    print("Unique Values per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

# Example: count_unique_values(load_data("../data/customers.csv"))
# Output: Unique Values per Column:
# ID: 1000 unique values
# Name: 950 unique values
# Age: 50 unique values
# Salary: 800 unique values
# City: 20 unique values

"""8. Memory Usage Analysis
Functions to analyze memory usage of a dataset."""
def analyze_memory_usage(df):
    """
    Calculate and report the memory usage of the dataset.
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is None:
        print("No data to analyze")
        return
    print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Example: analyze_memory_usage(load_data("../data/customers.csv"))
# Output: Memory Usage: 0.04 MB

"""9. Data Loading and Overview Exercises
Exercise 1: Load and Display Shape
Write a function that loads a dataset and displays its shape."""
def load_and_display_shape(file_path):
    """
    Load a dataset and print its shape.
    Args:
        file_path (str): Path to the data file
    """
    df = load_data(file_path)
    if df is not None:
        print(f"Shape: {df.shape}")

load_and_display_shape("../data/customers.csv")
# Output: Successfully loaded ../data/customers.csv
# Shape: (1000, 5)

"""Exercise 2: Count Numerical Columns
Write a function that counts numerical columns in a dataset."""
def count_numerical_columns(df):
    """
    Count the number of numerical columns in a dataset.
    Args:
        df (pandas.DataFrame): Input dataset
    Returns:
        int: Number of numerical columns
    """
    if df is None:
        return 0
    return len(df.select_dtypes(include=['int64', 'float64']).columns)

df = load_data("../data/customers.csv")
print(f"Numerical Columns: {count_numerical_columns(df)}")
# Output: Numerical Columns: 2

"""Exercise 3: List Categorical Columns
Write a function that lists categorical columns in a dataset."""
def list_categorical_columns(df):
    """
    List columns with object data type (categorical).
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is not None:
        categorical = df.select_dtypes(include=['object']).columns
        print(f"Categorical Columns: {list(categorical)}")

list_categorical_columns(load_data("../data/customers.csv"))
# Output: Categorical Columns: ['Name', 'City']

"""Exercise 4: Check Memory Threshold
Write a function that checks if memory usage exceeds a threshold."""
def check_memory_threshold(df, threshold_mb=1.0):
    """
    Check if dataset memory usage exceeds a threshold.
    Args:
        df (pandas.DataFrame): Input dataset
        threshold_mb (float): Memory threshold in MB
    """
    if df is None:
        print("No data to analyze")
        return
    memory = df.memory_usage().sum() / 1024**2
    print(f"Memory Usage: {memory:.2f} MB")
    if memory > threshold_mb:
        print("Warning: Large dataset detected!")
    else:
        print("Dataset size is within threshold.")

check_memory_threshold(load_data("../data/customers.csv"), 1.0)
# Output: Memory Usage: 0.04 MB
# Dataset size is within threshold.

"""Exercise 5: Missing Values Percentage
Write a function that reports missing values as percentages."""
def missing_values_percentage(df):
    """
    Report missing values as percentages per column.
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is None:
        print("No data to analyze")
        return
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found")
    else:
        print("Missing Values (%):")
        print(missing)

missing_values_percentage(load_data("../data/customers.csv"))
# Output: No missing values found

"""Exercise 6: Suggest Data Type Optimization
Write a function that suggests data type optimizations."""
def suggest_data_type_optimization(df):
    """
    Suggest data type optimizations for memory efficiency.
    Args:
        df (pandas.DataFrame): Input dataset
    """
    if df is None:
        print("No data to analyze")
        return
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0 and df[col].max() <= 255:
            print(f"Column {col} (int64) could be converted to uint8")
    for col in df.select_dtypes(include=['float64']).columns:
        if df[col].dropna().apply(lambda x: x.is_integer()).all():
            print(f"Column {col} (float64) could be converted to int")

suggest_data_type_optimization(load_data("../data/customers.csv"))
# Output: (Depends on data; e.g., no output if no optimizations possible)

"""Exercise 7: Load Multiple Files
Write a function that loads multiple files into a list of DataFrames."""
def load_multiple_files(file_paths):
    """
    Load multiple files into a list of DataFrames.
    Args:
        file_paths (list): List of file paths
    Returns:
        list: List of loaded DataFrames
    """
    return [load_data(path) for path in file_paths]

file_paths = ["../data/customers.csv", "../data/orders.csv"]
dfs = load_multiple_files(file_paths)
# Output: Successfully loaded ../data/customers.csv
# Successfully loaded ../data/orders.csv

"""Exercise 8: High Unique Values Check
Write a function that identifies columns with high unique values."""
def high_unique_values(df, threshold=100):
    """
    Identify columns with unique values exceeding a threshold.
    Args:
        df (pandas.DataFrame): Input dataset
        threshold (int): Unique values threshold
    """
    if df is None:
        print("No data to analyze")
        return
    print(f"Columns with >{threshold} unique values:")
    for col in df.columns:
        if df[col].nunique() > threshold:
            print(f"{col}: {df[col].nunique()} unique values")

high_unique_values(load_data("../data/customers.csv"), 100)
# Output: Columns with >100 unique values:
# ID: 1000 unique values
# Name: 950 unique values
# Salary: 800 unique values