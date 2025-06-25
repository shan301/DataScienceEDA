# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:06:28 2025

@author: Shantanu
"""

"""Data Cleaning
This module provides utility functions for cleaning datasets, a critical step in exploratory data analysis (EDA) for data science workflows. It covers handling missing values, duplicates, outliers, and data type conversions.

1. Data Loading
Functions to load data from CSV files."""
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the data file
    Returns:
        pandas.DataFrame: Loaded dataset or None if loading fails
    """
    try:
        if Path(file_path).suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {file_path}")
            return df
        else:
            raise ValueError("Only CSV files are supported")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

load_data("../data/customers.csv")  # Output: Successfully loaded ../data/customers.csv

"""2. Missing Values Detection
Functions to identify and summarize missing values."""
def detect_missing_values(df):
    """
    Summarize missing values by column.
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

# Example: detect_missing_values(load_data("../data/customers.csv"))
# Output: No missing values found

"""3. Handle Missing Values
Functions to handle missing values using imputation or deletion."""
def impute_missing_values(df, strategy='mean', columns=None):
    """
    Impute missing values in numerical or categorical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        strategy (str): Imputation strategy ('mean', 'median', 'mode', or 'drop')
        columns (list, optional): Columns to impute
    Returns:
        pandas.DataFrame: Dataset with imputed values
    """
    if df is None:
        print("No data to process")
        return None
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.columns
    for col in columns:
        if col in df_copy.columns:
            if strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
            elif df_copy[col].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                elif strategy == 'median':
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif strategy == 'mode':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    return df_copy

# Example: impute_missing_values(load_data("../data/customers.csv"), strategy='mean', columns=['Salary'])
# Output: (Returns DataFrame with imputed Salary values)

"""4. Duplicate Detection and Removal
Functions to identify and remove duplicate rows."""
def remove_duplicates(df):
    """
    Identify and remove duplicate rows.
    Args:
        df (pandas.DataFrame): Input dataset
    Returns:
        pandas.DataFrame: Dataset without duplicates
    """
    if df is None:
        print("No data to process")
        return None
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate rows")
    df_clean = df.drop_duplicates()
    print(f"Removed duplicates, remaining rows: {len(df_clean)}")
    return df_clean

# Example: remove_duplicates(load_data("../data/customers.csv"))
# Output: Found 0 duplicate rows
# Removed duplicates, remaining rows: 1000

"""5. Outlier Detection
Functions to detect outliers using IQR method."""
def detect_outliers_iqr(df, column):
    """
    Detect outliers in a numerical column using IQR method.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    Returns:
        pandas.Series: Boolean mask for outliers
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return None
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    print(f"Found {outliers.sum()} outliers in {column}")
    return outliers

# Example: detect_outliers_iqr(load_data("../data/customers.csv"), "Salary")
# Output: Found X outliers in Salary (depends on data)

"""6. Outlier Handling
Functions to handle outliers by capping or removing them."""
def handle_outliers(df, column, method='cap'):
    """
    Handle outliers by capping or removing them.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        method (str): Handling method ('cap' or 'remove')
    Returns:
        pandas.DataFrame: Dataset with handled outliers
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return None
    df_copy = df.copy()
    Q1 = df_copy[column].quantile(0.25)
    Q3 = df_copy[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    if method == 'cap':
        df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
        print(f"Capped outliers in {column}")
    elif method == 'remove':
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
        print(f"Removed outliers in {column}, remaining rows: {len(df_copy)}")
    return df_copy

# Example: handle_outliers(load_data("../data/customers.csv"), "Salary", method='cap')
# Output: Capped outliers in Salary

"""7. Data Type Conversion
Functions to convert columns to appropriate data types."""
def convert_data_types(df, type_dict):
    """
    Convert columns to specified data types.
    Args:
        df (pandas.DataFrame): Input dataset
        type_dict (dict): Dictionary mapping column names to target types
    Returns:
        pandas.DataFrame: Dataset with converted types
    """
    if df is None:
        print("No data to process")
        return None
    df_copy = df.copy()
    for col, dtype in type_dict.items():
        if col in df_copy.columns:
            try:
                df_copy[col] = df_copy[col].astype(dtype)
                print(f"Converted {col} to {dtype}")
            except Exception as e:
                print(f"Error converting {col}: {e}")
    return df_copy

# Example: convert_data_types(load_data("../data/customers.csv"), {'Age': 'int32', 'Salary': 'float32'})
# Output: Converted Age to int32
# Converted Salary to float32

"""8. String Cleaning
Functions to clean string columns (e.g., strip whitespace, standardize case)."""
def clean_strings(df, columns):
    """
    Clean string columns by stripping whitespace and standardizing case.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list): List of string columns to clean
    Returns:
        pandas.DataFrame: Dataset with cleaned strings
    """
    if df is None:
        print("No data to process")
        return None
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].str.strip().str.lower()
            print(f"Cleaned strings in {col}")
    return df_copy

# Example: clean_strings(load_data("../data/customers.csv"), ['Name', 'City'])
# Output: Cleaned strings in Name
# Cleaned strings in City

"""9. Data Cleaning Exercises
Exercise 1: Detect Missing Values Percentage
Write a function to report missing values as percentages."""
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

"""Exercise 2: Impute with Median
Write a function to impute missing values with median for numerical columns."""
def impute_median(df, columns=None):
    """
    Impute missing values with median for numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to impute
    Returns:
        pandas.DataFrame: Dataset with imputed values
    """
    return impute_missing_values(df, strategy='median', columns=columns)

df = load_data("../data/customers.csv")
impute_median(df, ['Salary'])
# Output: (Returns DataFrame with median-imputed Salary)

"""Exercise 3: Remove Rows with Missing Values
Write a function to remove rows with missing values in specified columns."""
def drop_missing_rows(df, columns=None):
    """
    Remove rows with missing values in specified columns.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to check
    Returns:
        pandas.DataFrame: Dataset without missing rows
    """
    return impute_missing_values(df, strategy='drop', columns=columns)

drop_missing_rows(load_data("../data/customers.csv"), ['Age'])
# Output: (Returns DataFrame with rows dropped if Age is missing)

"""Exercise 4: Count Duplicates
Write a function to count duplicate rows."""
def count_duplicates(df):
    """
    Count duplicate rows in a dataset.
    Args:
        df (pandas.DataFrame): Input dataset
    Returns:
        int: Number of duplicate rows
    """
    if df is None:
        print("No data to analyze")
        return 0
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    return duplicates

count_duplicates(load_data("../data/customers.csv"))
# Output: Duplicate rows: 0

"""Exercise 5: Remove Outliers by Removal
Write a function to remove outliers using IQR method."""
def remove_outliers_iqr(df, column):
    """
    Remove outliers in a column using IQR method.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    Returns:
        pandas.DataFrame: Dataset without outliers
    """
    return handle_outliers(df, column, method='remove')

remove_outliers_iqr(load_data("../data/customers.csv"), "Salary")
# Output: Removed outliers in Salary, remaining rows: X (depends on data)

"""Exercise 6: Cap Outliers
Write a function to cap outliers using IQR method."""
def cap_outliers_iqr(df, column):
    """
    Cap outliers in a column using IQR method.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    Returns:
        pandas.DataFrame: Dataset with capped outliers
    """
    return handle_outliers(df, column, method='cap')

cap_outliers_iqr(load_data("../data/customers.csv"), "Salary")
# Output: Capped outliers in Salary

"""Exercise 7: Convert to Categorical
Write a function to convert a column to categorical type."""
def convert_to_categorical(df, column):
    """
    Convert a column to categorical data type.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    Returns:
        pandas.DataFrame: Dataset with converted column
    """
    return convert_data_types(df, {column: 'category'})

convert_to_categorical(load_data("../data/customers.csv"), "City")
# Output: Converted City to category

"""Exercise 8: Clean Whitespace
Write a function to clean whitespace from string columns."""
def clean_whitespace(df, columns):
    """
    Remove leading/trailing whitespace from string columns.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list): List of string columns
    Returns:
        pandas.DataFrame: Dataset with cleaned strings
    """
    return clean_strings(df, columns)

clean_whitespace(load_data("../data/customers.csv"), ['Name'])
# Output: Cleaned strings in Name

"""Exercise 9: Standardize Numerical Column
Write a function to standardize a numerical column (zero mean, unit variance)."""
def standardize_column(df, column):
    """
    Standardize a numerical column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    Returns:
        pandas.DataFrame: Dataset with standardized column
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return None
    df_copy = df.copy()
    df_copy[column] = (df_copy[column] - df_copy[column].mean()) / df_copy[column].std()
    print(f"Standardized {column}")
    return df_copy

standardize_column(load_data("../data/customers.csv"), "Salary")
# Output: Standardized Salary

"""Exercise 10: Replace Invalid Values
Write a function to replace invalid values with NaN."""
def replace_invalid(df, column, invalid_values):
    """
    Replace invalid values in a column with NaN.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        invalid_values (list): Values to replace with NaN
    Returns:
        pandas.DataFrame: Dataset with replaced values
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return None
    df_copy = df.copy()
    df_copy[column] = df_copy[column].replace(invalid_values, np.nan)
    print(f"Replaced invalid values in {column}")
    return df_copy

replace_invalid(load_data("../data/customers.csv"), "Age", [-1, 999])
# Output: Replaced invalid values in Age

"""Exercise 11: Check Data Type Consistency
Write a function to check if a column's values match its data type."""
def check_type_consistency(df, column):
    """
    Check if a column's values are consistent with its data type.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    dtype = df[column].dtype
    if dtype in ['int64', 'float64']:
        non_numeric = pd.to_numeric(df[column], errors='coerce').isna() & df[column].notna()
        if non_numeric.sum() > 0:
            print(f"Found {non_numeric.sum()} non-numeric values in {column}")
        else:
            print(f"All values in {column} are consistent with {dtype}")
    else:
        print(f"Column {column} is of type {dtype}, no numeric check performed")

check_type_consistency(load_data("../data/customers.csv"), "Age")
# Output: All values in Age are consistent with int64

"""Exercise 12: Impute with Mode for Categorical
Write a function to impute missing values with mode for categorical columns."""
def impute_mode_categorical(df, columns=None):
    """
    Impute missing values with mode for categorical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to impute
    Returns:
        pandas.DataFrame: Dataset with imputed values
    """
    return impute_missing_values(df, strategy='mode', columns=columns)

impute_mode_categorical(load_data("../data/customers.csv"), ['City'])
# Output: (Returns DataFrame with mode-imputed City)

"""Exercise 13: Remove Columns with High Missing Values
Write a function to remove columns with missing values above a threshold."""
def remove_high_missing_columns(df, threshold=0.5):
    """
    Remove columns with missing value percentage above threshold.
    Args:
        df (pandas.DataFrame): Input dataset
        threshold (float): Missing value threshold (0 to 1)
    Returns:
        pandas.DataFrame: Dataset with columns removed
    """
    if df is None:
        print("No data to process")
        return None
    df_copy = df.copy()
    missing = df_copy.isnull().mean()
    columns_to_drop = missing[missing > threshold].index
    df_copy = df_copy.drop(columns=columns_to_drop)
    if len(columns_to_drop) > 0:
        print(f"Dropped columns: {list(columns_to_drop)}")
    else:
        print("No columns dropped")
    return df_copy

remove_high_missing_columns(load_data("../data/customers.csv"), 0.5)
# Output: No columns dropped

"""Exercise 14: Clean Numerical Outliers with Z-Score
Write a function to remove outliers using Z-score method."""
def remove_outliers_zscore(df, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        threshold (float): Z-score threshold
    Returns:
        pandas.DataFrame: Dataset without outliers
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return None
    df_copy = df.copy()
    z_scores = np.abs((df_copy[column] - df_copy[column].mean()) / df_copy[column].std())
    df_copy = df_copy[z_scores < threshold]
    print(f"Removed outliers in {column}, remaining rows: {len(df_copy)}")
    return df_copy

remove_outliers_zscore(load_data("../data/customers.csv"), "Salary")
# Output: Removed outliers in Salary, remaining rows: X (depends on data)

"""Exercise 15: Validate Column Values
Write a function to validate values in a column against a range."""
def validate_column_range(df, column, min_val, max_val):
    """
    Validate if column values are within a specified range.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        min_val (float): Minimum allowed value
        max_val (float): Maximum allowed value
    Returns:
        pandas.DataFrame: Dataset with invalid values replaced with NaN
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return None
    df_copy = df.copy()
    invalid = (df_copy[column] < min_val) | (df_copy[column] > max_val)
    if invalid.sum() > 0:
        df_copy.loc[invalid, column] = np.nan
        print(f"Replaced {invalid.sum()} invalid values in {column} with NaN")
    else:
        print(f"All values in {column} are within range")
    return df_copy

validate_column_range(load_data("../data/customers.csv"), "Age", 18, 100)
# Output: All values in Age are within range