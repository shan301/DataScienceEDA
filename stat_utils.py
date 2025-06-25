# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 21:23:30 2025

@author: Shantanu
"""
"""Statistical Utilities
This script provides utility functions for statistical analysis in Exploratory Data Analysis (EDA). It includes functions for descriptive statistics, hypothesis testing, outlier detection, and correlation analysis, compatible with pandas, Dask, and Vaex. Designed for datasets like orders.csv, products.csv, employees.csv, sales.csv, and financials.csv.
"""

import pandas as pd
import dask.dataframe as dd
import vaex
import numpy as np
from scipy import stats
from data_utils import load_dataset

"""1. Descriptive Statistics
Compute descriptive statistics for numerical columns."""
def get_descriptive_stats(df, columns=None, library='pandas'):
    """Compute descriptive statistics (mean, median, std, min, max, etc.) for numerical columns.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        columns (list): Numerical columns to analyze. If None, uses all numeric columns.
        library (str): Library of the DataFrame ('pandas', 'dask', 'vaex').
    
    Returns:
        DataFrame: Descriptive statistics.
    """
    if columns is None:
        if library == 'pandas':
            columns = df.select_dtypes(include=[np.number]).columns
        elif library == 'dask':
            columns = df.select_dtypes(include=[np.number]).columns
        elif library == 'vaex':
            columns = df.get_column_names(numeric=True)
        else:
            raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")
    
    if library == 'pandas':
        stats_df = df[columns].describe()
    elif library == 'dask':
        stats_df = df[columns].describe().compute()
    elif library == 'vaex':
        stats_dict = {col: df[col].describe() for col in columns}
        stats_df = pd.DataFrame(stats_dict)
    return stats_df

# Example: Descriptive stats for Amount in orders.csv
orders_df = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
stats_orders = get_descriptive_stats(orders_df, columns=['Amount'])
print(stats_orders)  # Output: Descriptive statistics for Amount

"""2. Hypothesis Test (T-Test)
Perform an independent t-test between two groups."""
def perform_ttest(df, group_col, value_col, group1, group2, library='pandas'):
    """Perform an independent t-test between two groups for a numerical column.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        group_col (str): Categorical column defining groups.
        value_col (str): Numerical column to test.
        group1 (str): First group value.
        group2 (str): Second group value.
        library (str): Library of the DataFrame ('pandas', 'dask', 'vaex').
    
    Returns:
        tuple: (t-statistic, p-value).
    """
    if library == 'pandas':
        data = df
    elif library == 'dask':
        data = df.compute()
    elif library == 'vaex':
        data = df.to_pandas_df()
    else:
        raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")
    
    group1_data = data[data[group_col] == group1][value_col].dropna()
    group2_data = data[data[group_col] == group2][value_col].dropna()
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    return t_stat, p_value

# Example: T-test for Salary between Sales and IT in employees.csv
employees_df = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
t_stat, p_val = perform_ttest(employees_df, group_col='Department', value_col='Salary', group1='Sales', group2='IT')
print(f"T-statistic: {t_stat}, P-value: {p_val}")  # Output: T-test results

"""3. Outlier Detection
Detect outliers using the IQR method."""
def detect_outliers(df, column, library='pandas'):
    """Detect outliers in a numerical column using the IQR method.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        column (str): Numerical column to analyze.
        library (str): Library of the DataFrame ('pandas', 'dask', 'vaex').
    
    Returns:
        DataFrame: Rows containing outliers.
    """
    if library == 'pandas':
        data = df
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
    elif library == 'dask':
        data = df
        q1 = data[column].quantile(0.25).compute()
        q3 = data[column].quantile(0.75).compute()
    elif library == 'vaex':
        data = df
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
    else:
        raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")
    
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if library == 'pandas':
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    elif library == 'dask':
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].compute()
    elif library == 'vaex':
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].to_pandas_df()
    return outliers

# Example: Detect outliers in Price in products.csv
products_df = load_dataset('data/products.csv', library='pandas')
outliers_products = detect_outliers(products_df, column='Price')
print(outliers_products.head())  # Output: Rows with outlier prices

"""4. Correlation Analysis
Compute correlations between numerical columns."""
def get_correlations(df, columns=None, method='pearson', library='pandas'):
    """Compute correlations between numerical columns.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        columns (list): Numerical columns to include. If None, uses all numeric columns.
        method (str): Correlation method ('pearson', 'spearman'). Default: 'pearson'.
        library (str): Library to use for loading the DataFrame ('pandas', 'dask', 'vaex').
    
    Returns:
        pd.DataFrame: Correlation matrix.
    """
    if columns is None:
        if library == 'pandas':
            columns = df.select_dtypes(include=[np.number]).columns
        elif library == 'dask':
            columns = df.select_dtypes(include=[np.number]).columns
        elif library == 'vaex':
            columns = df.get_column_names(numeric=True)
        else:
            raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")

    if library == 'pandas':
        corr_matrix = df[columns].corr(method=method)
    elif library == 'dask':
        corr_matrix = df[columns].corr(method=method).compute()
    elif library == 'vaex':
        data = df.to_pandas_df(columns)
        corr_matrix = data.corr(method=method)
    return corr_matrix

# Example: Correlation between Revenue and Expenses in financials.csv
financials_df = load_dataset('data/financials.csv', library='pandas', parse_dates=['Date'])
corr_financials = get_correlations(financials_df, columns=['Revenue', 'Expenses'])
print(corr_financials)  # Output: Correlation matrix

"""Exercises
Practice statistical analysis with the following exercises using orders.csv, products.csv, employees.csv, and financials.csv."""

"""Exercise 1: Descriptive Stats for Salary
Compute descriptive statistics for Salary in employees.csv using pandas."""
emp_ex1 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
stats_ex1 = get_descriptive_stats(emp_ex1, columns=['Salary'])
print(stats_ex1)  # Output: Descriptive statistics for Salary

"""Exercise 2: Descriptive Stats with Dask
Compute descriptive statistics for Amount in orders.csv using Dask."""
orders_ex2 = load_dataset('data/orders.csv', library='dask', parse_dates=['Date'])
stats_ex2 = get_descriptive_stats(orders_ex2, columns=['Amount'], library='dask')
print(stats_ex2)  # Output: Descriptive statistics for Amount

"""Exercise 3: Descriptive Stats with Vaex
Compute descriptive statistics for Price in products.csv using Vaex."""
products_ex3 = load_dataset('data/products.csv', library='vaex')
stats_ex3 = get_descriptive_stats(products_ex3, columns=['Price'], library='vaex')
print(stats_ex3)  # Output: Descriptive statistics for Price

"""Exercise 4: T-Test for Amount
Perform a t-test for Amount between North and South regions in orders.csv using pandas."""
orders_ex4 = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
t_stat_ex4, p_val_ex4 = perform_ttest(orders_ex4, group_col='Region', value_col='Amount', group1='North', group2='South')
print(f"T-statistic: {t_stat_ex4}, P-value: {p_val_ex4}")  # Output: T-test results

"""Exercise 5: T-Test with Dask
Perform a t-test for Revenue between two groups in financials.csv using Dask (e.g., split by year)."""
financials_ex5 = load_dataset('data/financials.csv', library='dask', parse_dates=['Date'])
financials_ex5['Year'] = financials_ex5['Date'].dt.year
t_stat_ex5, p_val_ex5 = perform_ttest(financials_ex5, group_col='Year', value_col='Revenue', group1=2022, group2=2023, library='dask')
print(f"T-statistic: {t_stat_ex5}, P-value: {p_val_ex5}")  # Output: T-test results

"""Exercise 6: Outlier Detection for Salary
Detect outliers in Salary in employees.csv using pandas."""
emp_ex6 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
outliers_ex6 = detect_outliers(emp_ex6, column='Salary')
print(outliers_ex6.head())  # Output: Rows with outlier salaries

"""Exercise 7: Outlier Detection with Vaex
Detect outliers in Price in products.csv using Vaex."""
products_ex7 = load_dataset('data/products.csv', library='vaex')
outliers_ex7 = detect_outliers(products_ex7, column='Price', library='vaex')
print(outliers_ex7.head())  # Output: Rows with outlier prices

"""Exercise 8: Correlation Analysis for Products
Compute Pearson correlations for Price and Stock in products.csv using pandas."""
products_ex8 = load_dataset('data/products.csv', library='pandas')
corr_ex8 = get_correlations(products_ex8, columns=['Price', 'Stock'])
print(corr_ex8)  # Output: Correlation matrix

"""Exercise 9: Correlation with Dask
Compute Spearman correlations for Revenue and Expenses in financials.csv using Dask."""
financials_ex9 = load_dataset('data/financials.csv', library='dask', parse_dates=['Date'])
corr_ex9 = get_correlations(financials_ex9, columns=['Revenue', 'Expenses'], method='spearman', library='dask')
print(corr_ex9)  # Output: Correlation matrix

"""Exercise 10: Descriptive Stats for Revenue
Compute descriptive statistics for Revenue in financials.csv using pandas."""
financials_ex10 = load_dataset('data/financials.csv', library='pandas', parse_dates=['Date'])
stats_ex10 = get_descriptive_stats(financials_ex10, columns=['Revenue'])
print(stats_ex10)  # Output: Descriptive statistics for Revenue

"""Exercise 11: T-Test for Stock
Perform a t-test for Stock between Electronics and Clothing categories in products.csv using pandas."""
products_ex11 = load_dataset('data/products.csv', library='pandas')
t_stat_ex11, p_val_ex11 = perform_ttest(products_ex11, group_col='Category', value_col='Stock', group1='Electronics', group2='Clothing')
print(f"T-statistic: {t_stat_ex11}, P-value: {p_val_ex11}")  # Output: T-test results

"""Exercise 12: Outlier Detection with Dask
Detect outliers in Amount in orders.csv using Dask."""
orders_ex12 = load_dataset('data/orders.csv', library='dask', parse_dates=['Date'])
outliers_ex12 = detect_outliers(orders_ex12, column='Amount', library='dask')
print(outliers_ex12.head())  # Output: Rows with outlier amounts

"""Exercise 13: Correlation for Employees
Compute Pearson correlations for Salary and EmployeeID in employees.csv using pandas."""
emp_ex13 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
corr_ex13 = get_correlations(emp_ex13, columns=['Salary', 'EmployeeID'])
print(corr_ex13)  # Output: Correlation matrix

"""Exercise 14: Combined Stats and Test
Compute descriptive stats for Salary and perform a t-test between HR and Finance in employees.csv."""
emp_ex14 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
stats_ex14 = get_descriptive_stats(emp_ex14, columns=['Salary'])
print("Descriptive Stats:\n", stats_ex14)
t_stat_ex14, p_val_ex14 = perform_ttest(emp_ex14, group_col='Department', value_col='Salary', group1='HR', group2='Finance')
print(f"T-statistic: {t_stat_ex14}, P-value: {p_val_ex14}")  # Output: Stats and t-test results

"""Exercise 15: Outlier Detection for Revenue
Detect outliers in Revenue in financials.csv using pandas."""
financials_ex15 = load_dataset('data/financials.csv', library='pandas', parse_dates=['Date'])
outliers_ex15 = detect_outliers(financials_ex15, column='Revenue')
print(outliers_ex15.head())  # Output: Rows with outlier revenues

"""Notes
- Ensure datasets (orders.csv, products.csv, employees.csv, sales.csv, financials.csv) are in the data/ directory.
- Install required libraries: `pip install pandas numpy scipy dask vaex`.
- Functions rely on data_utils.py for loading datasets (ensure it exists in advanced/).
- Use visualization_utils.py for plotting results (e.g., plot outliers or correlations).
- For large datasets, Dask and Vaex require .compute() or .to_pandas_df() for some operations.
- Hypothesis tests assume normality; verify assumptions for real-world data.
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass
