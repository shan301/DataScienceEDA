# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 12:56:47 2025

@author: Shantanu
"""

"""Basic Visualizations
This module provides utility functions for creating basic visualizations (histograms, box plots, scatter plots) to support exploratory data analysis (EDA) in data science workflows.

1. Setup and Data Loading
Functions to set up plotting environment and load data."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def setup_plotting():
    """
    Set up plotting environment with consistent style.
    """
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    print("Plotting environment configured")

setup_plotting()  # Output: Plotting environment configured

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

"""2. Histogram
Functions to create histograms for numerical columns."""
def plot_histogram(df, column, bins=30):
    """
    Create a histogram for a numerical column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name to plot
        bins (int): Number of bins
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    plt.figure()
    sns.histplot(data=df, x=column, bins=bins)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Example: plot_histogram(load_data("../data/customers.csv"), "Age")
# Output: (Displays histogram of Age column)

"""3. Box Plot
Functions to create box plots for numerical columns."""
def plot_box(df, column):
    """
    Create a box plot for a numerical column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name to plot
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    plt.figure()
    sns.boxplot(data=df, y=column)
    plt.title(f"Box Plot of {column}")
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()

# Example: plot_box(load_data("../data/customers.csv"), "Salary")
# Output: (Displays box plot of Salary column)

"""4. Scatter Plot
Functions to create scatter plots for two numerical columns."""
def plot_scatter(df, x_column, y_column, hue=None):
    """
    Create a scatter plot for two numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
        hue (str, optional): Column name for color coding
    """
    if df is None or x_column not in df.columns or y_column not in df.columns:
        print("Invalid data or columns")
        return
    plt.figure()
    sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue)
    plt.title(f"Scatter Plot of {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()

# Example: plot_scatter(load_data("../data/customers.csv"), "Age", "Salary", hue="City")
# Output: (Displays scatter plot of Age vs Salary, colored by City)

"""5. Pair Plot
Functions to create pair plots for multiple numerical columns."""
def plot_pair(df, columns=None):
    """
    Create a pair plot for numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): List of column names to include
    """
    if df is None:
        print("Invalid data")
        return
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(columns) < 2:
        print("Need at least two numerical columns")
        return
    sns.pairplot(df[columns])
    plt.suptitle("Pair Plot of Numerical Columns", y=1.02)
    plt.tight_layout()
    plt.show()

# Example: plot_pair(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: (Displays pair plot of Age and Salary)

"""6. Basic Visualizations Exercises
Exercise 1: Histogram with Custom Bins
Write a function to plot a histogram with a specified number of bins."""
def custom_histogram(df, column, bins=20):
    """
    Plot a histogram with a custom number of bins.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        bins (int): Number of bins
    """
    plot_histogram(df, column, bins)

df = load_data("../data/customers.csv")
custom_histogram(df, "Age", 20)
# Output: (Displays histogram of Age with 20 bins)

"""Exercise 2: Box Plot for Multiple Columns
Write a function to create box plots for multiple numerical columns."""
def multi_box_plot(df, columns=None):
    """
    Create box plots for multiple numerical columns.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): List of column names
    """
    if df is None:
        print("Invalid data")
        return
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure()
    sns.boxplot(data=df[columns])
    plt.title("Box Plots of Numerical Columns")
    plt.tight_layout()
    plt.show()

multi_box_plot(load_data("../data/customers.csv"), ["Age", "Salary"])
# Output: (Displays box plots for Age and Salary)

"""Exercise 3: Scatter Plot with Size
Write a function to create a scatter plot with point sizes based on a column."""
def scatter_with_size(df, x_column, y_column, size_column):
    """
    Create a scatter plot with point sizes based on a column.
    Args:
        df (pandas.DataFrame): Input dataset
        x_column (str): X-axis column
        y_column (str): Y-axis column
        size_column (str): Column for point sizes
    """
    if df is None or any(col not in df.columns for col in [x_column, y_column, size_column]):
        print("Invalid data or columns")
        return
    plt.figure()
    sns.scatterplot(data=df, x=x_column, y=y_column, size=size_column)
    plt.title(f"Scatter Plot of {x_column} vs {y_column} (Size: {size_column})")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()

scatter_with_size(load_data("../data/customers.csv"), "Age", "Salary", "Salary")
# Output: (Displays scatter plot with point sizes based on Salary)

"""Exercise 4: Histogram by Category
Write a function to plot histograms for a numerical column by category."""
def histogram_by_category(df, column, category):
    """
    Plot histograms for a numerical column, split by a categorical column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Numerical column name
        category (str): Categorical column name
    """
    if df is None or column not in df.columns or category not in df.columns:
        print("Invalid data or columns")
        return
    plt.figure()
    sns.histplot(data=df, x=column, hue=category, multiple="stack")
    plt.title(f"Histogram of {column} by {category}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

histogram_by_category(load_data("../data/customers.csv"), "Salary", "City")
# Output: (Displays stacked histogram of Salary by City)

"""Exercise 5: Box Plot by Category
Write a function to create box plots for a numerical column by category."""
def box_plot_by_category(df, column, category):
    """
    Create box plots for a numerical column, split by a categorical column.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Numerical column name
        category (str): Categorical column name
    """
    if df is None or column not in df.columns or category not in df.columns:
        print("Invalid data or columns")
        return
    plt.figure()
    sns.boxplot(data=df, x=category, y=column)
    plt.title(f"Box Plot of {column} by {category}")
    plt.xlabel(category)
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

box_plot_by_category(load_data("../data/customers.csv"), "Salary", "City")
# Output: (Displays box plots of Salary by City)

"""Exercise 6: Save Visualization
Write a function to save a histogram to a file."""
def save_histogram(df, column, bins=30, output_path="histogram.png"):
    """
    Save a histogram to a file.
    Args:
        df (pandas.DataFrame): Input dataset
        column (str): Column name
        bins (int): Number of bins
        output_path (str): Path to save the plot
    """
    if df is None or column not in df.columns:
        print("Invalid data or column")
        return
    plt.figure()
    sns.histplot(data=df, x=column, bins=bins)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved histogram to {output_path}")

save_histogram(load_data("../data/customers.csv"), "Age", output_path="age_histogram.png")
# Output: Saved histogram to age_histogram.png

"""Exercise 7: Scatter Plot with Regression Line
Write a function to create a scatter plot with a regression line."""
def scatter_with_regression(df, x_column, y_column):
    """
    Create a scatter plot with a regression line.
    Args:
        df (pandas.DataFrame): Input dataset
        x_column (str): X-axis column
        y_column (str): Y-axis column
    """
    if df is None or x_column not in df.columns or y_column not in df.columns:
        print("Invalid data or columns")
        return
    plt.figure()
    sns.regplot(data=df, x=x_column, y=y_column)
    plt.title(f"Scatter Plot of {x_column} vs {y_column} with Regression Line")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()

scatter_with_regression(load_data("../data/customers.csv"), "Age", "Salary")
# Output: (Displays scatter plot with regression line)

"""Exercise 8: Custom Pair Plot
Write a function to create a pair plot with custom styling."""
def custom_pair_plot(df, columns=None, hue=None):
    """
    Create a pair plot with custom styling.
    Args:
        df (pandas.DataFrame): Input dataset
        columns (list, optional): List of column names
        hue (str, optional): Column for color coding
    """
    if df is None:
        print("Invalid data")
        return
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(columns) < 2:
        print("Need at least two numerical columns")
        return
    sns.pairplot(df, vars=columns, hue=hue, diag_kind="hist")
    plt.suptitle("Custom Pair Plot", y=1.02)
    plt.tight_layout()
    plt.show()

custom_pair_plot(load_data("../data/customers.csv"), ["Age", "Salary"], hue="City")
# Output: (Displays pair plot with histograms on diagonal, colored by City)