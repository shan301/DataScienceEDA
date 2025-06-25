# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:13:42 2025

@author: Shantanu
"""

"""Correlation Analysis
This module provides utility functions for performing correlation analysis, a key component of exploratory data analysis (EDA) in data science workflows. It covers Pearson, Spearman, and Kendall correlations, correlation matrices, and visualizations like heatmaps.

1. Data Loading
Functions to load data from CSV files."""
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
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

"""2. Pearson Correlation
Functions to compute Pearson correlation coefficient for linear relationships."""
def pearson_correlation(df, column1, column2):
    """
    Compute Pearson correlation between two numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
    """
    if df is None or column1 not in df.columns or column2 not in df.columns:
        print("Invalid data or columns")
        return
    corr, p_value = stats.pearsonr(df[column1], df[column2])
    print(f"Pearson Correlation between {column1} and {column2}:")
    print(f"Correlation: {corr:.3f}")
    print(f"P-value: {p_value:.3f}")

# Example: pearson_correlation(load_data("../data/customers.csv"), "Age", "Salary")
# Output: Pearson Correlation between Age and Salary:
# Correlation: 0.650
# P-value: 0.000 (depends on data)

"""3. Spearman Correlation
Functions to compute Spearman correlation for monotonic relationships."""
def spearman_correlation(df, column1, column2):
    """
    Compute Spearman correlation between two numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
    """
    if df is None or column1 not in df.columns or column2 not in df.columns:
        print("Invalid data or columns")
        return
    corr, p_value = stats.spearmanr(df[column1], df[column2])
    print(f"Spearman Correlation between {column1} and {column2}:")
    print(f"Correlation: {corr:.3f}")
    print(f"P-value: {p_value:.3f}")

# Example: spearman_correlation(load_data("../data/customers.csv"), "Age", "Salary")
# Output: Spearman Correlation between Age and Salary:
# Correlation: 0.620
# P-value: 0.000 (depends on data)

"""4. Kendall Correlation
Functions to compute Kendall correlation for ordinal data."""
def kendall_correlation(df, column1, column2):
    """
    Compute Kendall correlation between two numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
    """
    if df is None or column1 not in df.columns or column2 not in df.columns:
        print("Invalid data or columns")
        return
    corr, p_value = stats.kendalltau(df[column1], df[column2])
    print(f"Kendall Correlation between {column1} and {column2}:")
    print(f"Correlation: {corr:.3f}")
    print(f"P-value: {p_value:.3f}")

# Example: kendall_correlation(load_data("../data/customers.csv"), "Age", "Salary")
# Output: Kendall Correlation between Age and Salary:
# Correlation: 0.450
# P-value: 0.000 (depends on data)

"""5. Correlation Matrix
Functions to compute correlation matrix for multiple numerical columns."""
def correlation_matrix(df, columns=None, method='pearson'):
    """
    Compute correlation matrix for numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
        method (str): Correlation method ('pearson', 'spearman', 'kendall')
    Returns:
        pandas.DataFrame: Correlation matrix
    """
    if df is None:
        print("Invalid data")
        return None
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(columns) < 2:
        print("Need at least two numerical columns")
        return None
    corr_matrix = df[columns].corr(method=method)
    print(f"{method.capitalize()} Correlation Matrix:")
    print(corr_matrix)
    return corr_matrix

# Example: correlation_matrix(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: Pearson Correlation Matrix:
#            Age    Salary
# Age     1.000    0.650
# Salary  0.650    1.000 (depends on data)

"""6. Correlation Heatmap
Functions to visualize correlation matrix as a heatmap."""
def correlation_heatmap(df, columns=None, method='pearson'):
    """
    Visualize correlation matrix as a heatmap.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
        method (str): Correlation method ('pearson', 'spearman', 'kendall')
    """
    if df is None:
        print("Invalid data")
        return
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(columns) < 2:
        print("Need at least two numerical columns")
        return
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[columns].corr(method=method), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"{method.capitalize()} Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# Example: correlation_heatmap(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: (Displays heatmap with correlations annotated)

"""7. Correlation Analysis Exercises
Exercise 1: Compute Pearson Correlation
Write a function to compute Pearson correlation between two columns."""
def compute_pearson(df, column1, column2):
    """
    Compute Pearson correlation between two columns.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
    """
    pearson_correlation(df, column1, column2)

compute_pearson(load_data("../data/customers.csv"), "Age", "Salary")
# Output: Pearson Correlation between Age and Salary:
# Correlation: 0.650
# P-value: 0.000 (depends on data)

"""Exercise 2: Compute Spearman Correlation
Write a function to compute Spearman correlation between two columns."""
def compute_spearman(df, column1, column2):
    """
    Compute Spearman correlation between two columns.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
    """
    spearman_correlation(df, column1, column2)

compute_spearman(load_data("../data/customers.csv"), "Age", "Salary")
# Output: Spearman Correlation between Age and Salary:
# Correlation: 0.620
# P-value: 0.000 (depends on data)

"""Exercise 3: Compute Kendall Correlation
Write a function to compute Kendall correlation between two columns."""
def compute_kendall(df, column1, column2):
    """
    Compute Kendall correlation between two columns.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
    """
    kendall_correlation(df, column1, column2)

compute_kendall(load_data("../data/customers.csv"), "Age", "Salary")
# Output: Kendall Correlation between Age and Salary:
# Correlation: 0.450
# P-value: 0.000 (depends on data)

"""Exercise 4: Generate Pearson Correlation Matrix
Write a function to compute Pearson correlation matrix."""
def pearson_matrix(df, columns=None):
    """
    Compute Pearson correlation matrix.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
    """
    correlation_matrix(df, columns, method='pearson')

pearson_matrix(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: Pearson Correlation Matrix:
#            Age    Salary
# Age     1.000    0.650
# Salary  0.650    1.000 (depends on data)

"""Exercise 5: Generate Spearman Correlation Matrix
Write a function to compute Spearman correlation matrix."""
def spearman_matrix(df, columns=None):
    """
    Compute Spearman correlation matrix.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
    """
    correlation_matrix(df, columns, method='spearman')

spearman_matrix(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: Spearman Correlation Matrix:
#            Age    Salary
# Age     1.000    0.620
# Salary  0.620    1.000 (depends on data)

"""Exercise 6: Generate Kendall Correlation Matrix
Write a function to compute Kendall correlation matrix."""
def kendall_matrix(df, columns=None):
    """
    Compute Kendall correlation matrix.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
    """
    correlation_matrix(df, columns, method='kendall')

kendall_matrix(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: Kendall Correlation Matrix:
#            Age    Salary
# Age     1.000    0.450
# Salary  0.450    1.000 (depends on data)

"""Exercise 7: Visualize Pearson Heatmap
Write a function to visualize Pearson correlation heatmap."""
def pearson_heatmap(df, columns=None):
    """
    Visualize Pearson correlation heatmap.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
    """
    correlation_heatmap(df, columns, method='pearson')

pearson_heatmap(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: (Displays Pearson correlation heatmap)

"""Exercise 8: Visualize Spearman Heatmap
Write a function to visualize Spearman correlation heatmap."""
def spearman_heatmap(df, columns=None):
    """
    Visualize Spearman correlation heatmap.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
    """
    correlation_heatmap(df, columns, method='spearman')

spearman_heatmap(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: (Displays Spearman correlation heatmap)

"""Exercise 9: Compare Correlation Methods
Write a function to compare Pearson, Spearman, and Kendall correlations."""
def compare_correlations(df, column1, column2):
    """
    Compare Pearson, Spearman, and Kendall correlations.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
    """
    if df is None or column1 not in df.columns or column2 not in df.columns:
        print("Invalid data or columns")
        return
    pearson_corr, _ = stats.pearsonr(df[column1], df[column2])
    spearman_corr, _ = stats.spearmanr(df[column1], df[column2])
    kendall_corr, _ = stats.kendalltau(df[column1], df[column2])
    print(f"Correlation Comparison for {column1} and {column2}:")
    print(f"Pearson: {pearson_corr:.3f}")
    print(f"Spearman: {spearman_corr:.3f}")
    print(f"Kendall: {kendall_corr:.3f}")

compare_correlations(load_data("../data/customers.csv"), "Age", "Salary")
# Output: Correlation Comparison for Age and Salary:
# Pearson: 0.650
# Spearman: 0.620
# Kendall: 0.450 (depends on data)

"""Exercise 10: Identify Strong Correlations
Write a function to identify correlations above a threshold."""
def strong_correlations(df, threshold=0.7, method='pearson'):
    """
    Identify correlations above a threshold.
    Args:
        df (pandas.DataFrame): Input dataset
        threshold (float): Correlation threshold
        method (str): Correlation method
    """
    corr_matrix = correlation_matrix(df, method=method)
    if corr_matrix is None:
        return
    strong = corr_matrix.abs().unstack().sort_values(ascending=False)
    strong = strong[(strong > threshold) & (strong < 1.0)]
    print(f"Strong {method.capitalize()} Correlations (|corr| > {threshold}):")
    print(strong)

strong_correlations(load_data("../data/customers.csv"), threshold=0.7)
# Output: Strong Pearson Correlations (|corr| > 0.7):
# (Depends on data)

"""Exercise 11: Correlation with P-value Filter
Write a function to filter correlations by p-value."""
def significant_correlations(df, columns=None, p_threshold=0.05, method='pearson'):
    """
    Identify significant correlations based on p-value.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
        p_threshold (float): P-value threshold
        method (str): Correlation method ('pearson', 'spearman')
    """
    if df is None:
        print("Invalid data")
        return
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(columns) < 2:
        print("Need at least two numerical columns")
        return
    print(f"Significant {method.capitalize()} Correlations (p < {p_threshold}):")
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            if method == 'pearson':
                corr, p = stats.pearsonr(df[col1], df[col2])
            elif method == 'spearman':
                corr, p = stats.spearmanr(df[col1], df[col2])
            else:
                continue  # Kendall not included for simplicity
            if p < p_threshold:
                print(f"{col1} vs {col2}: Corr={corr:.3f}, P-value={p:.3f}")

significant_correlations(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: Significant Pearson Correlations (p < 0.05):
# Age vs Salary: Corr=0.650, P-value=0.000 (depends on data)

"""Exercise 12: Save Correlation Heatmap
Write a function to save a correlation heatmap to a file."""
def save_correlation_heatmap(df, columns=None, method='pearson', output_path="heatmap.png"):
    """
    Save correlation heatmap to a file.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
        method (str): Correlation method
        output_path (str): Path to save the plot
    """
    if df is None:
        print("Invalid data")
        return
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(columns) < 2:
        print("Need at least two numerical columns")
        return
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[columns].corr(method=method), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"{method.capitalize()} Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap to {output_path}")

save_correlation_heatmap(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: Saved heatmap to heatmap.png

"""Exercise 13: Partial Correlation
Write a function to compute partial correlation controlling for a third variable."""
def partial_correlation(df, column1, column2, control_column):
    """
    Compute partial correlation between two columns, controlling for a third.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
        control_column (str): Control column name
    """
    if df is None or any(col not in df.columns for col in [column1, column2, control_column]):
        print("Invalid data or columns")
        return
    # Compute residuals after regressing on control column
    def residuals(y, x):
        slope, intercept = np.polyfit(x, y, 1)
        return y - (slope * x + intercept)
    res1 = residuals(df[column1], df[control_column])
    res2 = residuals(df[column2], df[control_column])
    corr, p_value = stats.pearsonr(res1, res2)
    print(f"Partial Correlation between {column1} and {column2} (controlling {control_column}):")
    print(f"Correlation: {corr:.3f}")
    print(f"P-value: {p_value:.3f}")

partial_correlation(load_data("../data/customers.csv"), "Age", "Salary", "ID")
# Output: Partial Correlation between Age and Salary (controlling ID):
# Correlation: 0.640
# P-value: 0.000 (depends on data)

"""Exercise 14: Correlation Matrix with Mask
Write a function to display correlation matrix with upper triangle masked."""
def masked_correlation_heatmap(df, columns=None, method='pearson'):
    """
    Visualize correlation matrix with upper triangle masked.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): Columns to include
        method (str): Correlation method
    """
    if df is None:
        print("Invalid data")
        return
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(columns) < 2:
        print("Need at least two numerical columns")
        return
    corr_matrix = df[columns].corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"{method.capitalize()} Correlation Heatmap (Masked)")
    plt.tight_layout()
    plt.show()

masked_correlation_heatmap(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: (Displays masked heatmap)

"""Exercise 15: Correlation Significance Test
Write a function to test if a correlation is statistically significant."""
def test_correlation_significance(df, column1, column2, method='pearson', alpha=0.05):
    """
    Test if correlation is statistically significant.
    Args:
        df (pandas.DataFrame): Input dataset
        column1 (str): First column name
        column2 (str): Second column name
        method (str): Correlation method ('pearson', 'spearman')
        alpha (float): Significance level
    """
    if df is None or column1 not in df.columns or column2 not in df.columns:
        print("Invalid data or columns")
        return
    if method == 'pearson':
        corr, p_value = stats.pearsonr(df[column1], df[column2])
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(df[column1], df[column2])
    else:
        print("Unsupported method")
        return
    print(f"{method.capitalize()} Correlation Test for {column1} vs {column2}:")
    print(f"Correlation: {corr:.3f}, P-value: {p_value:.3f}")
    if p_value < alpha:
        print("Correlation is statistically significant")
    else:
        print("Correlation is not statistically significant")

test_correlation_significance(load_data("../data/customers.csv"), "Age", "Salary")
# Output: Pearson Correlation Test for Age vs Salary:
# Correlation: 0.650, P-value: 0.000
# Correlation is statistically significant (depends on data)