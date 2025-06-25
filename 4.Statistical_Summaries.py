# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:12:03 2025

@author: Shantanu
"""

"""Statistical Summaries
This module provides utility functions for generating statistical summaries, crucial for exploratory data analysis (EDA) in data science workflows. It covers measures of central tendency, dispersion, distribution characteristics, and skewness/kurtosis.

1. Data Loading
Functions to load data from CSV files."""
import pandas as pd
import numpy as np
from scipy import stats
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

"""2. Basic Descriptive Statistics
Functions to compute basic descriptive statistics for numerical columns."""
def basic_descriptive_stats(df, columns=None):
    """
    Compute basic descriptive statistics (mean, median, std, min, max).
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to analyze
    """
    if df is None:
        print("No data to analyze")
        return
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    stats_df = df[columns].describe()
    print("Basic Descriptive Statistics:")
    print(stats_df)

# Example: basic_descriptive_stats(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: Basic Descriptive Statistics:
# (Shows count, mean, std, min, 25%, 50%, 75%, max for Age, Salary)

"""3. Measures of Central Tendency
Functions to compute mean, median, and mode."""
def central_tendency(df, column):
    """
    Compute mean, median, and mode for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]
    print(f"Central Tendency for {column}:")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode:.2f}")

# Example: central_tendency(load_data("../data/customers.csv"), "Age")
# Output: Central Tendency for Age:
# Mean: 35.50
# Median: 34.00
# Mode: 30.00 (depends on data)

"""4. Measures of Dispersion
Functions to compute variance, standard deviation, and range."""
def dispersion_measures(df, column):
    """
    Compute variance, standard deviation, and range for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    variance = df[column].var()
    std_dev = df[column].std()
    data_range = df[column].max() - df[column].min()
    print(f"Dispersion Measures for {column}:")
    print(f"Variance: {variance:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Range: {data_range:.2f}")

# Example: dispersion_measures(load_data("../data/customers.csv"), "Salary")
# Output: Dispersion Measures for Salary:
# Variance: 25000000.00
# Standard Deviation: 5000.00
# Range: 20000.00 (depends on data)

"""5. Percentiles and Quartiles
Functions to compute percentiles and quartiles."""
def percentiles_quartiles(df, column, percentiles=[25, 50, 75]):
    """
    Compute specified percentiles and quartiles for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        percentiles (list): List of percentiles to compute
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    perc_values = df[column].quantile([p/100 for p in percentiles])
    print(f"Percentiles for {column}:")
    for p, val in zip(percentiles, perc_values):
        print(f"{p}th Percentile: {val:.2f}")

# Example: percentiles_quartiles(load_data("../data/customers.csv"), "Salary")
# Output: Percentiles for Salary:
# 25th Percentile: 45000.00
# 50th Percentile: 60000.00
# 75th Percentile: 75000.00 (depends on data)

"""6. Skewness and Kurtosis
Functions to compute skewness and kurtosis to assess distribution shape."""
def skewness_kurtosis(df, column):
    """
    Compute skewness and kurtosis for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    skewness = df[column].skew()
    kurtosis = df[column].kurtosis()
    print(f"Distribution Measures for {column}:")
    print(f"Skewness: {skewness:.2f}")
    print(f"Kurtosis: {kurtosis:.2f}")

# Example: skewness_kurtosis(load_data("../data/customers.csv"), "Salary")
# Output: Distribution Measures for Salary:
# Skewness: 0.50
# Kurtosis: -0.20 (depends on data)

"""7. Frequency Tables for Categorical Data
Functions to generate frequency tables for categorical columns."""
def frequency_table(df, column):
    """
    Generate a frequency table for a categorical column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    freq = df[column].value_counts()
    print(f"Frequency Table for {column}:")
    print(freq)

# Example: frequency_table(load_data("../data/customers.csv"), "City")
# Output: Frequency Table for City:
# New York    200
# Chicago     150
# ... (depends on data)

"""8. Statistical Summaries Exercises
Exercise 1: Compute Mean and Median
Write a function to compute mean and median for a numerical column."""
def compute_mean_median(df, column):
    """
    Compute mean and median for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    print(f"Mean: {df[column].mean():.2f}")
    print(f"Median: {df[column].median():.2f}")

compute_mean_median(load_data("../data/customers.csv"), "Age")
# Output: Mean: 35.50
# Median: 34.00 (depends on data)

"""Exercise 2: Compute Standard Deviation
Write a function to compute standard deviation for a numerical column."""
def compute_std_dev(df, column):
    """
    Compute standard deviation for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    print(f"Standard Deviation: {df[column].std():.2f}")

compute_std_dev(load_data("../data/customers.csv"), "Salary")
# Output: Standard Deviation: 5000.00 (depends on data)

"""Exercise 3: Compute IQR
Write a function to compute the interquartile range (IQR) for a column."""
def compute_iqr(df, column):
    """
    Compute IQR for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    iqr = Q3 - Q1
    print(f"IQR for {column}: {iqr:.2f}")

compute_iqr(load_data("../data/customers.csv"), "Salary")
# Output: IQR for Salary: 30000.00 (depends on data)

"""Exercise 4: Compute Mode
Write a function to compute the mode for a column."""
def compute_mode(df, column):
    """
    Compute mode for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    mode = df[column].mode()[0]
    print(f"Mode for {column}: {mode}")

compute_mode(load_data("../data/customers.csv"), "City")
# Output: Mode for City: New York (depends on data)

"""Exercise 5: Compute Variance
Write a function to compute variance for a numerical column."""
def compute_variance(df, column):
    """
    Compute variance for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    print(f"Variance: {df[column].var():.2f}")

compute_variance(load_data("../data/customers.csv"), "Age")
# Output: Variance: 100.00 (depends on data)

"""Exercise 6: Compute Custom Percentiles
Write a function to compute custom percentiles for a column."""
def compute_custom_percentiles(df, column, percentiles):
    """
    Compute custom percentiles for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        percentiles (list): List of percentiles
    """
    percentiles_quartiles(df, column, percentiles)

compute_custom_percentiles(load_data("../data/customers.csv"), "Salary", [10, 90])
# Output: Percentiles for Salary:
# 10th Percentile: 35000.00
# 90th Percentile: 85000.00 (depends on data)

"""Exercise 7: Compute Skewness
Write a function to compute skewness for a numerical column."""
def compute_skewness(df, column):
    """
    Compute skewness for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    print(f"Skewness: {df[column].skew():.2f}")

compute_skewness(load_data("../data/customers.csv"), "Salary")
# Output: Skewness: 0.50 (depends on data)

"""Exercise 8: Compute Kurtosis
Write a function to compute kurtosis for a numerical column."""
def compute_kurtosis(df, column):
    """
    Compute kurtosis for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    print(f"Kurtosis: {df[column].kurtosis():.2f}")

compute_kurtosis(load_data("../data/customers.csv"), "Salary")
# Output: Kurtosis: -0.20 (depends on data)

"""Exercise 9: Frequency Table with Proportions
Write a function to generate a frequency table with proportions."""
def frequency_table_proportions(df, column):
    """
    Generate a frequency table with proportions for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    freq = df[column].value_counts(normalize=True) * 100
    print(f"Frequency Table with Proportions (%) for {column}:")
    print(freq)

frequency_table_proportions(load_data("../data/customers.csv"), "City")
# Output: Frequency Table with Proportions (%) for City:
# New York    20.0
# Chicago     15.0
# ... (depends on data)

"""Exercise 10: Compare Central Tendency
Write a function to compare mean, median, and mode for a column."""
def compare_central_tendency(df, column):
    """
    Compare mean, median, and mode for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    central_tendency(df, column)

compare_central_tendency(load_data("../data/customers.csv"), "Age")
# Output: Central Tendency for Age:
# Mean: 35.50
# Median: 34.00
# Mode: 30.00 (depends on data)

"""Exercise 11: Compute Coefficient of Variation
Write a function to compute the coefficient of variation for a column."""
def coefficient_of_variation(df, column):
    """
    Compute coefficient of variation (std/mean) for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    cv = df[column].std() / df[column].mean()
    print(f"Coefficient of Variation for {column}: {cv:.2f}")

coefficient_of_variation(load_data("../data/customers.csv"), "Salary")
# Output: Coefficient of Variation for Salary: 0.08 (depends on data)

"""Exercise 12: Median Absolute Deviation
Write a function to compute median absolute deviation (MAD)."""
def median_absolute_deviation(df, column):
    """
    Compute MAD for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    print(f"MAD for {column}: {mad:.2f}")

median_absolute_deviation(load_data("../data/customers.csv"), "Age")
# Output: MAD for Age: 5.00 (depends on data)

"""Exercise 13: Statistical Summary for Multiple Columns
Write a function to compute summary stats for multiple columns."""
def multi_column_stats(df, columns=None):
    """
    Compute descriptive statistics for multiple columns.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to analyze
    """
    basic_descriptive_stats(df, columns)

multi_column_stats(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: Basic Descriptive Statistics:
# (Shows stats for Age, Salary)

"""Exercise 14: Trimmed Mean
Write a function to compute trimmed mean for a column."""
def trimmed_mean(df, column, proportion=0.1):
    """
    Compute trimmed mean for a column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        proportion (float): Proportion to trim from each tail
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    trimmed = stats.trim_mean(df[column].dropna(), proportion)
    print(f"Trimmed Mean for {column}: {trimmed:.2f}")

trimmed_mean(load_data("../data/customers.csv"), "Salary", 0.1)
# Output: Trimmed Mean for Salary: 60000.00 (depends on data)

"""Exercise 15: Detect Outliers with Z-Score
Write a function to detect outliers using Z-score."""
def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-score method.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        threshold (float): Z-score threshold
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = z_scores > threshold
    print(f"Outliers in {column}: {outliers.sum()}")

detect_outliers_zscore(load_data("../data/customers.csv"), "Salary")
# Output: Outliers in Salary: X (depends on data)