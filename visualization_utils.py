# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 21:22:25 2025

@author: Shantanu
"""

```python
"""Visualization Utilities
This script provides utility functions for creating interactive visualizations in Exploratory Data Analysis (EDA). It includes functions for histograms, scatter plots, box plots, and correlation heatmaps using Plotly, compatible with pandas, Dask, and Vaex. Designed for datasets like orders.csv, products.csv, employees.csv, sales.csv, and financials.csv.
"""

import pandas as pd
import dask.dataframe as dd
import vaex
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from data_utils import load_dataset

"""1. Histogram
Create a histogram for a numerical column."""
def plot_histogram(df, column, title=None, library='pandas'):
    """Plot a histogram for a numerical column.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        column (str): Column to plot.
        title (str): Plot title. If None, defaults to 'Histogram of {column}'.
        library (str): Library of the DataFrame ('pandas', 'dask', 'vaex').
    
    Returns:
        None: Displays an interactive Plotly histogram.
    """
    if title is None:
        title = f"Histogram of {column}"
    
    if library == 'pandas':
        data = df[column]
    elif library == 'dask':
        data = df[column].compute()
    elif library == 'vaex':
        data = df[column].to_pandas_series()
    else:
        raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")
    
    fig = px.histogram(x=data, title=title, labels={column: column}, nbins=30)
    fig.update_layout(xaxis_title=column, yaxis_title='Count')
    fig.show()

# Example: Histogram of Amount in orders.csv
orders_df = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
plot_histogram(orders_df, column='Amount', title='Distribution of Order Amounts')  # Output: Interactive histogram

"""2. Scatter Plot
Create a scatter plot for two numerical columns."""
def plot_scatter(df, x_col, y_col, color_col=None, title=None, library='pandas'):
    """Plot a scatter plot for two numerical columns.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        x_col (str): X-axis column.
        y_col (str): Y-axis column.
        color_col (str): Optional column for color encoding.
        title (str): Plot title. If None, defaults to 'Scatter Plot of {x_col} vs {y_col}'.
        library (str): Library of the DataFrame ('pandas', 'dask', 'vaex').
    
    Returns:
        None: Displays an interactive Plotly scatter plot.
    """
    if title is None:
        title = f"Scatter Plot of {x_col} vs {y_col}"
    
    if library == 'pandas':
        data = df
    elif library == 'dask':
        data = df.compute()
    elif library == 'vaex':
        data = df.to_pandas_df()
    else:
        raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")
    
    fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=title)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    fig.show()

# Example: Scatter plot of Price vs Stock in products.csv
products_df = load_dataset('data/products.csv', library='pandas')
plot_scatter(products_df, x_col='Price', y_col='Stock', color_col='Category')  # Output: Interactive scatter plot

"""3. Box Plot
Create a box plot for a numerical column grouped by a categorical column."""
def plot_box(df, x_col, y_col, title=None, library='pandas'):
    """Plot a box plot for a numerical column grouped by a categorical column.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        x_col (str): Categorical column for grouping.
        y_col (str): Numerical column for box plot.
        title (str): Plot title. If None, defaults to 'Box Plot of {y_col} by {x_col}'.
        library (str): Library of the DataFrame ('pandas', 'dask', 'vaex').
    
    Returns:
        None: Displays an interactive Plotly box plot.
    """
    if title is None:
        title = f"Box Plot of {y_col} by {x_col}"
    
    if library == 'pandas':
        data = df
    elif library == 'dask':
        data = df.compute()
    elif library == 'vaex':
        data = df.to_pandas_df()
    else:
        raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")
    
    fig = px.box(data, x=x_col, y=y_col, title=title)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    fig.show()

# Example: Box plot of Salary by Department in employees.csv
employees_df = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
plot_box(employees_df, x_col='Department', y_col='Salary')  # Output: Interactive box plot

"""4. Correlation Heatmap
Create a correlation heatmap for numerical columns."""
def plot_correlation_heatmap(df, columns=None, title='Correlation Heatmap', library='pandas'):
    """Plot a correlation heatmap for numerical columns.
    
    Args:
        df: Input DataFrame (pandas, Dask, or Vaex).
        columns (list): Numerical columns to include. If None, uses all numeric columns.
        title (str): Plot title.
        library (str): Library of the DataFrame ('pandas', 'dask', 'vaex').
    
    Returns:
        None: Displays an interactive Plotly heatmap.
    """
    if library == 'pandas':
        data = df
    elif library == 'dask':
        data = df.compute()
    elif library == 'vaex':
        data = df.to_pandas_df()
    else:
        raise ValueError("Library must be 'pandas', 'dask', or 'vaex'.")
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    corr_matrix = data[columns].corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale='Viridis',
        annotation_text=corr_matrix.round(2).values
    )
    fig.update_layout(title=title, xaxis_title='Columns', yaxis_title='Columns')
    fig.show()

# Example: Correlation heatmap for numerical columns in financials.csv
financials_df = load_dataset('data/financials.csv', library='pandas', parse_dates=['Date'])
plot_correlation_heatmap(financials_df, columns=['Revenue', 'Expenses'])  # Output: Interactive heatmap

"""Exercises
Practice visualization techniques with the following exercises using orders.csv, products.csv, employees.csv, and financials.csv."""

"""Exercise 1: Histogram of Salary
Plot a histogram of Salary in employees.csv using pandas."""
emp_ex1 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
plot_histogram(emp_ex1, column='Salary', title='Employee Salary Distribution')  # Output: Interactive histogram

"""Exercise 2: Histogram with Dask
Plot a histogram of Amount in orders.csv using Dask."""
orders_ex2 = load_dataset('data/orders.csv', library='dask', parse_dates=['Date'])
plot_histogram(orders_ex2, column='Amount', library='dask')  # Output: Interactive histogram

"""Exercise 3: Histogram with Vaex
Plot a histogram of Price in products.csv using Vaex."""
products_ex3 = load_dataset('data/products.csv', library='vaex')
plot_histogram(products_ex3, column='Price', library='vaex')  # Output: Interactive histogram

"""Exercise 4: Scatter Plot of Orders
Plot a scatter plot of Amount vs ProductID in orders.csv, colored by Region."""
orders_ex4 = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
plot_scatter(orders_ex4, x_col='ProductID', y_col='Amount', color_col='Region')  # Output: Interactive scatter plot

"""Exercise 5: Scatter Plot with Dask
Plot a scatter plot of Revenue vs Expenses in financials.csv using Dask."""
financials_ex5 = load_dataset('data/financials.csv', library='dask', parse_dates=['Date'])
plot_scatter(financials_ex5, x_col='Revenue', y_col='Expenses', library='dask')  # Output: Interactive scatter plot

"""Exercise 6: Box Plot of Amount by Region
Plot a box plot of Amount by Region in orders.csv using pandas."""
orders_ex6 = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
plot_box(orders_ex6, x_col='Region', y_col='Amount')  # Output: Interactive box plot

"""Exercise 7: Box Plot with Vaex
Plot a box plot of Salary by Department in employees.csv using Vaex."""
emp_ex7 = load_dataset('data/employees.csv', library='vaex', parse_dates=['HireDate'])
plot_box(emp_ex7, x_col='Department', y_col='Salary', library='vaex')  # Output: Interactive box plot

"""Exercise 8: Correlation Heatmap for Products
Plot a correlation heatmap for Price and Stock in products.csv using pandas."""
products_ex8 = load_dataset('data/products.csv', library='pandas')
plot_correlation_heatmap(products_ex8, columns=['Price', 'Stock'])  # Output: Interactive heatmap

"""Exercise 9: Correlation Heatmap with Dask
Plot a correlation heatmap for Revenue and Expenses in financials.csv using Dask."""
financials_ex9 = load_dataset('data/financials.csv', library='dask', parse_dates=['Date'])
plot_correlation_heatmap(financials_ex9, columns=['Revenue', 'Expenses'], library='dask')  # Output: Interactive heatmap

"""Exercise 10: Histogram of Revenue
Plot a histogram of Revenue in financials.csv using pandas."""
financials_ex10 = load_dataset('data/financials.csv', library='pandas', parse_dates=['Date'])
plot_histogram(financials_ex10, column='Revenue')  # Output: Interactive histogram

"""Exercise 11: Scatter Plot with Categories
Plot a scatter plot of Price vs Stock in products.csv, colored by Category, using pandas."""
products_ex11 = load_dataset('data/products.csv', library='pandas')
plot_scatter(products_ex11, x_col='Price', y_col='Stock', color_col='Category')  # Output: Interactive scatter plot

"""Exercise 12: Box Plot of Amount by ProductID
Plot a box plot of Amount by ProductID in orders.csv using pandas (limit to top 5 ProductIDs for clarity)."""
orders_ex12 = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
top_products = orders_ex12['ProductID'].value_counts().head(5).index
orders_ex12 = orders_ex12[orders_ex12['ProductID'].isin(top_products)]
plot_box(orders_ex12, x_col='ProductID', y_col='Amount')  # Output: Interactive box plot

"""Exercise 13: Correlation Heatmap for Employees
Plot a correlation heatmap for Salary and EmployeeID in employees.csv using pandas."""
emp_ex13 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
plot_correlation_heatmap(emp_ex13, columns=['Salary', 'EmployeeID'])  # Output: Interactive heatmap

"""Exercise 14: Combined Scatter and Box Plot
Plot a scatter plot of Amount vs ProductID and a box plot of Amount by Region in orders.csv."""
orders_ex14 = load_dataset('data/orders.csv', library='pandas', parse_dates=['Date'])
plot_scatter(orders_ex14, x_col='ProductID', y_col='Amount', color_col='Region')
plot_box(orders_ex14, x_col='Region', y_col='Amount')  # Output: Two interactive plots

"""Exercise 15: Histogram with Custom Title
Plot a histogram of Salary in employees.csv with a custom title 'Employee Salary Distribution 2023'."""
emp_ex15 = load_dataset('data/employees.csv', library='pandas', parse_dates=['HireDate'])
plot_histogram(emp_ex15, column='Salary', title='Employee Salary Distribution 2023')  # Output: Interactive histogram

"""Notes
- Ensure datasets (orders.csv, products.csv, employees.csv, sales.csv, financials.csv) are in the data/ directory.
- Install required libraries: `pip install pandas dask vaex plotly`.
- Functions rely on data_utils.py for loading datasets (ensure it exists in advanced/).
- Use stats_utils.py for statistical analysis (not included here).
- For large datasets, Dask and Vaex require .compute() or .to_pandas_df() for Plotly compatibility.
- Visualizations are interactive; view in a Jupyter notebook or browser.
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass
```
