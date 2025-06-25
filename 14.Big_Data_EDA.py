# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:19:23 2025

@author: Shantanu
"""

"""Big Data EDA
Exploratory Data Analysis (EDA) for large datasets requires scalable tools to handle data that exceeds memory limits. This script covers EDA using Dask, Vaex, and Modin for efficient data processing and visualization, applied to sample datasets (sales.csv, financials.csv, orders.csv).
"""

# Import required libraries
import dask.dataframe as dd
import vaex
import modin.pandas as mpd
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

"""1. Loading Large Datasets with Dask
Dask loads data lazily, allowing out-of-memory computations."""
# Load sales.csv with Dask
sales_ddf = dd.read_csv('data/sales.csv', parse_dates=['Date'])
print(sales_ddf.head())  # Output: First 5 rows of sales.csv
print(sales_ddf.info())  # Output: Dask DataFrame structure and dtypes

"""2. Basic Summaries with Dask
Compute summary statistics for large datasets."""
# Compute descriptive statistics
sales_summary = sales_ddf.describe().compute()
print(sales_summary)  # Output: Summary statistics (count, mean, std, min, max, etc.)

"""3. GroupBy Operations with Dask
Perform groupby operations efficiently."""
# Group by Region and compute mean Amount
sales_grouped = sales_ddf.groupby('Region')['Amount'].mean().compute()
print(sales_grouped)  # Output: Mean Amount by Region

"""4. Visualizing Dask Results
Visualize aggregated results using Plotly."""
# Plot mean Amount by Region
fig = px.bar(x=sales_grouped.index, y=sales_grouped.values,
             title='Average Sales Amount by Region (Dask)',
             labels={'x': 'Region', 'y': 'Mean Amount'})
fig.update_layout(xaxis_title='Region', yaxis_title='Mean Amount')
fig.show()  # Output: Interactive bar plot

"""5. Loading Data with Vaex
Vaex is optimized for large datasets with fast groupby and joins."""
# Load financials.csv with Vaex
financials_vdf = vaex.from_csv('data/financials.csv', parse_dates=['Date'])
print(financials_vdf.head())  # Output: First 5 rows of financials.csv
print(financials_vdf.info())  # Output: Vaex DataFrame structure and dtypes

"""6. Statistical Analysis with Vaex
Compute statistics like mean, median, and correlations."""
# Compute mean and median for Revenue
revenue_stats = financials_vdf[['Revenue']].agg({'Revenue': ['mean', 'median']})
print(revenue_stats)  # Output: Mean and median of Revenue

"""7. Visualizing Vaex Results
Create histograms or scatter plots with Vaex."""
# Plot histogram of Revenue
financials_vdf.viz.histogram(financials_vdf.Revenue, title='Revenue Distribution (Vaex)')
plt.show()  # Output: Histogram of Revenue values

"""8. Using Modin for Pandas-like EDA
Modin distributes pandas operations for scalability."""
# Load orders.csv with Modin
orders_mdf = mpd.read_csv('data/orders.csv', parse_dates=['Date'])
print(orders_mdf.head())  # Output: First 5 rows of orders.csv
print(orders_mdf.info())  # Output: Modin DataFrame structure and dtypes

"""9. Filtering and Aggregations with Modin
Perform filtering and aggregations similar to pandas."""
# Filter orders with Amount > 1000 and compute total Amount by CustomerID
high_value_orders = orders_mdf[orders_mdf['Amount'] > 1000]
customer_totals = high_value_orders.groupby('CustomerID')['Amount'].sum()
print(customer_totals.head())  # Output: Total Amount for top 5 CustomerIDs

"""Exercises
Practice big data EDA techniques with the following exercises using sales.csv, financials.csv, and orders.csv."""

"""Exercise 1: Load Sales Data with Dask
Load sales.csv with Dask and display the first 10 rows."""
sales_ex1 = dd.read_csv('data/sales.csv', parse_dates=['Date'])
print(sales_ex1.head(10))  # Output: First 10 rows of sales.csv

"""Exercise 2: Dask Summary Statistics
Compute summary statistics for the Amount column in sales.csv using Dask."""
summary_ex2 = sales_ex1['Amount'].describe().compute()
print(summary_ex2)  # Output: Summary statistics for Amount

"""Exercise 3: Dask GroupBy
Group sales.csv by ProductID and compute the total Amount using Dask."""
grouped_ex3 = sales_ex1.groupby('ProductID')['Amount'].sum().compute()
print(grouped_ex3.head())  # Output: Total Amount for top 5 ProductIDs

"""Exercise 4: Visualize Dask GroupBy
Create a bar plot of total Amount by ProductID from Exercise 3."""
fig_ex4 = px.bar(x=grouped_ex3.index, y=grouped_ex3.values,
                 title='Total Sales Amount by ProductID (Dask)',
                 labels={'x': 'ProductID', 'y': 'Total Amount'})
fig_ex4.update_layout(xaxis_title='ProductID', yaxis_title='Total Amount')
fig_ex4.show()  # Output: Bar plot of total Amount by ProductID

"""Exercise 5: Load Financials with Vaex
Load financials.csv with Vaex and display the first 5 rows."""
financials_ex5 = vaex.from_csv('data/financials.csv', parse_dates=['Date'])
print(financials_ex5.head())  # Output: First 5 rows of financials.csv

"""Exercise 6: Vaex Descriptive Statistics
Compute mean, min, and max for Expenses in financials.csv using Vaex."""
stats_ex6 = financials_ex5[['Expenses']].agg({'Expenses': ['mean', 'min', 'max']})
print(stats_ex6)  # Output: Mean, min, and max of Expenses

"""Exercise 7: Vaex Histogram
Create a histogram of Expenses from financials.csv using Vaex."""
financials_ex5.viz.histogram(financials_ex5.Expenses, title='Expenses Distribution (Vaex)')
plt.show()  # Output: Histogram of Expenses

"""Exercise 8: Load Orders with Modin
Load orders.csv with Modin and display the first 10 rows."""
orders_ex8 = mpd.read_csv('data/orders.csv', parse_dates=['Date'])
print(orders_ex8.head(10))  # Output: First 10 rows of orders.csv

"""Exercise 9: Modin Filtering
Filter orders.csv with Modin to include only orders from 2023 and print the count."""
orders_ex9 = orders_ex8[orders_ex8['Date'].dt.year == 2023]
print(len(orders_ex9))  # Output: Number of orders in 2023

"""Exercise 10: Modin GroupBy
Group orders.csv with Modin by Region and compute the median Amount."""
median_ex10 = orders_ex8.groupby('Region')['Amount'].median()
print(median_ex10)  # Output: Median Amount by Region

"""Exercise 11: Dask Correlation Analysis
Compute the correlation between Amount and ProductID in sales.csv using Dask."""
corr_ex11 = sales_ex1[['Amount', 'ProductID']].corr().compute()
print(corr_ex11)  # Output: Correlation matrix for Amount and ProductID

"""Exercise 12: Vaex GroupBy
Group financials.csv by year (extracted from Date) and compute total Revenue using Vaex."""
financials_ex12 = financials_ex5.copy()
financials_ex12['Year'] = financials_ex12.Date.dt.year
grouped_ex12 = financials_ex12.groupby('Year', agg={'Revenue': 'sum'})
print(grouped_ex12)  # Output: Total Revenue by Year

"""Exercise 13: Visualize Vaex GroupBy
Create a bar plot of total Revenue by Year from Exercise 12."""
fig_ex13 = px.bar(grouped_ex12, x='Year', y='Revenue_sum',
                 title='Total Revenue by Year (Vaex)',
                 labels={'Revenue_sum': 'Total Revenue'})
fig_ex13.update_layout(xaxis_title='Year', yaxis_title='Total Revenue')
fig_ex13.show()  # Output: Bar plot of total Revenue by Year

"""Exercise 14: Modin Join
Join orders.csv and customers.csv on CustomerID using Modin and display the first 5 rows."""
customers_mdf = mpd.read_csv('data/customers.csv')
merged_ex14 = orders_ex8.merge(customers_mdf, on='CustomerID', how='inner')
print(merged_ex14.head())  # Output: First 5 rows of merged dataset

"""Exercise 15: Dask Missing Values
Compute the count of missing values in sales.csv using Dask."""
missing_ex15 = sales_ex1.isna().sum().compute()
print(missing_ex15)  # Output: Count of missing values per column

"""Notes
- Ensure datasets (sales.csv, financials.csv, orders.csv) have appropriate columns (e.g., Date, Amount, Region, Revenue, Expenses, OrderID, CustomerID).
- Install required libraries: `pip install dask vaex modin[ray] plotly`.
- Dask requires careful handling of `.compute()` to avoid memory overload.
- Vaex is optimized for read-only operations; use pandas for small datasets if modifications are needed.
- Modin uses Ray or Dask as a backend; ensure the backend is installed (e.g., `pip install ray`).
- For truly large datasets, test on files >1GB to observe performance benefits.
- Visualizations use Plotly for interactivity; ensure a compatible environment (e.g., Jupyter or browser).
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass