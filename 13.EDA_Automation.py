# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:16:27 2025

@author: Shantanu
"""

"""EDA Automation
Automated Exploratory Data Analysis (EDA) streamlines data exploration by generating comprehensive reports and visualizations. This script covers tools like pandas-profiling, Sweetviz, and D-Tale for automated EDA, applied to sample datasets (sales.csv, hr_data.csv, customers.csv).
"""

# Import required libraries
import pandas as pd
import sweetviz as sv
from ydata_profiling import ProfileReport
import dtale

"""1. Loading Sample Data
Load datasets for automated EDA (sales.csv, hr_data.csv, customers.csv)."""
# Load datasets
sales_df = pd.read_csv('data/sales.csv', parse_dates=['Date'])
hr_df = pd.read_csv('data/hr_data.csv')
customers_df = pd.read_csv('data/customers.csv')
print(sales_df.head())      # Output: First 5 rows of sales.csv
print(hr_df.head())         # Output: First 5 rows of hr_data.csv
print(customers_df.head())  # Output: First 5 rows of customers.csv

"""2. Pandas-Profiling Report
Pandas-Profiling generates a detailed HTML report with summaries, correlations, and distributions."""
# Generate pandas-profiling report for sales.csv
profile_sales = ProfileReport(sales_df, title="Sales Data Profiling Report")
profile_sales.to_file("sales_profiling_report.html")
print("Pandas-Profiling report generated: sales_profiling_report.html")  # Output: HTML report saved

"""3. Sweetviz Report
Sweetviz creates comparative EDA reports with visualizations for single or multiple datasets."""
# Generate Sweetviz report for hr_data.csv
hr_report = sv.analyze(hr_df)
hr_report.show_html("hr_sweetviz_report.html")
print("Sweetviz report generated: hr_sweetviz_report.html")  # Output: HTML report saved

"""4. Comparing Two Datasets with Sweetviz
Sweetviz can compare two datasets (e.g., training vs. test sets)."""
# Split customers.csv into train and test
train_cust, test_cust = customers_df.iloc[:int(len(customers_df)*0.8)], customers_df.iloc[int(len(customers_df)*0.8):]
compare_report = sv.compare([train_cust, "Train"], [test_cust, "Test"])
compare_report.show_html("customers_compare_report.html")
print("Sweetviz comparison report generated: customers_compare_report.html")  # Output: HTML report saved

"""5. D-Tale Interactive Exploration
D-Tale provides an interactive web interface for data exploration."""
# Launch D-Tale for hr_data.csv
d = dtale.show(hr_df, open_browser=False)
print(f"D-Tale URL: {d._url}")  # Output: URL for D-Tale interface (e.g., http://localhost:40000)

"""Exercises
Practice automated EDA techniques with the following exercises using sales.csv, hr_data.csv, and customers.csv."""

"""Exercise 1: Load and Inspect Data
Load sales.csv and display the first 10 rows."""
sales_ex1 = pd.read_csv('data/sales.csv', parse_dates=['Date'])
print(sales_ex1.head(10))  # Output: First 10 rows of sales.csv

"""Exercise 2: Pandas-Profiling for HR Data
Generate a pandas-profiling report for hr_data.csv."""
profile_ex2 = ProfileReport(hr_df, title="HR Data Profiling Report")
profile_ex2.to_file("hr_profiling_report.html")
print("Pandas-Profiling report generated: hr_profiling_report.html")  # Output: HTML report saved

"""Exercise 3: Minimal Pandas-Profiling Report
Generate a minimal pandas-profiling report for customers.csv (minimal=True)."""
profile_ex3 = ProfileReport(customers_df, title="Customers Minimal Profiling Report", minimal=True)
profile_ex3.to_file("customers_minimal_profiling_report.html")
print("Minimal Pandas-Profiling report generated: customers_minimal_profiling_report.html")  # Output: HTML report saved

"""Exercise 4: Sweetviz for Sales Data
Generate a Sweetviz report for sales.csv."""
sales_ex4 = sv.analyze(sales_df)
sales_ex4.show_html("sales_sweetviz_report.html")
print("Sweetviz report generated: sales_sweetviz_report.html")  # Output: HTML report saved

"""Exercise 5: Compare Subsets of Sales Data
Split sales.csv into two subsets (e.g., by Region) and generate a Sweetviz comparison report."""
sales_ex5_north = sales_df[sales_df['Region'] == 'North']
sales_ex5_south = sales_df[sales_df['Region'] == 'South']
compare_ex5 = sv.compare([sales_ex5_north, "North"], [sales_ex5_south, "South"])
compare_ex5.show_html("sales_region_compare_report.html")
print("Sweetviz comparison report generated: sales_region_compare_report.html")  # Output: HTML report saved

"""Exercise 6: D-Tale for Customers Data
Launch a D-Tale instance for customers.csv and print the URL."""
d_ex6 = dtale.show(customers_df, open_browser=False)
print(f"D-Tale URL: {d_ex6._url}")  # Output: URL for D-Tale interface

"""Exercise 7: Pandas-Profiling with Custom Title
Generate a pandas-profiling report for sales.csv with a custom title 'Sales EDA Report'."""
profile_ex7 = ProfileReport(sales_df, title="Sales EDA Report")
profile_ex7.to_file("sales_custom_profiling_report.html")
print("Pandas-Profiling report generated: sales_custom_profiling_report.html")  # Output: HTML report saved

"""Exercise 8: Sweetviz with Target Feature
Generate a Sweetviz report for customers.csv with Churn as the target feature."""
cust_ex8 = sv.analyze(customers_df, target_feat='Churn')
cust_ex8.show_html("customers_sweetviz_churn_report.html")
print("Sweetviz report generated: customers_sweetviz_churn_report.html")  # Output: HTML report saved

"""Exercise 9: D-Tale with Subset
Launch D-Tale for a subset of hr_data.csv where Salary > 50000."""
hr_ex9 = hr_df[hr_df['Salary'] > 50000]
d_ex9 = dtale.show(hr_ex9, open_browser=False)
print(f"D-Tale URL for high-salary subset: {d_ex9._url}")  # Output: URL for D-Tale interface

"""Exercise 10: Pandas-Profiling for Missing Values
Generate a pandas-profiling report for hr_data.csv and check the missing values section."""
profile_ex10 = ProfileReport(hr_df, title="HR Data Missing Values Report")
profile_ex10.to_file("hr_missing_profiling_report.html")
print("Pandas-Profiling report generated: hr_missing_profiling_report.html")  # Output: HTML report saved
# Note: Check the 'Missing Values' section in the HTML report

"""Exercise 11: Sweetviz Comparison by Date
Split sales.csv by year (e.g., before and after 2023) and generate a Sweetviz comparison report."""
sales_ex11 = sales_df.copy()
sales_ex11_before = sales_ex11[sales_ex11['Date'].dt.year < 2023]
sales_ex11_after = sales_ex11[sales_ex11['Date'].dt.year >= 2023]
compare_ex11 = sv.compare([sales_ex11_before, "Before 2023"], [sales_ex11_after, "2023 and After"])
compare_ex11.show_html("sales_year_compare_report.html")
print("Sweetviz comparison report generated: sales_year_compare_report.html")  # Output: HTML report saved

"""Exercise 12: D-Tale with Custom Port
Launch D-Tale for sales.csv on a specific port (e.g., 40001)."""
d_ex12 = dtale.show(sales_df, open_browser=False, port=40001)
print(f"D-Tale URL on port 40001: {d_ex12._url}")  # Output: URL for D-Tale interface

"""Exercise 13: Pandas-Profiling with Correlations
Generate a pandas-profiling report for customers.csv with only correlation analysis enabled."""
profile_ex13 = ProfileReport(customers_df, title="Customers Correlation Report", correlations={"auto": {"calculate": True}})
profile_ex13.to_file("customers_correlation_profiling_report.html")
print("Pandas-Profiling correlation report generated: customers_correlation_profiling_report.html")  # Output: HTML report saved

"""Exercise 14: Sweetviz with Custom Parameters
Generate a Sweetviz report for hr_data.csv with pairwise analysis disabled."""
hr_ex14 = sv.analyze(hr_df, pairwise_analysis='off')
hr_ex14.show_html("hr_sweetviz_no_pairwise_report.html")
print("Sweetviz report generated: hr_sweetviz_no_pairwise_report.html")  # Output: HTML report saved

"""Exercise 15: Combined D-Tale Analysis
Launch D-Tale for a concatenated dataset of sales.csv and customers.csv (merge on a common key if available)."""
# Assuming a common key (e.g., CustomerID) exists
merged_ex15 = pd.merge(sales_df, customers_df, on='CustomerID', how='inner')
d_ex15 = dtale.show(merged_ex15, open_browser=False)
print(f"D-Tale URL for merged dataset: {d_ex15._url}")  # Output: URL for D-Tale interface

"""Notes
- Ensure datasets (sales.csv, hr_data.csv, customers.csv) have appropriate columns (e.g., Date, Amount, Region, Age, Salary, Income, Churn).
- Install required libraries: `pip install pandas ydata-profiling sweetviz dtale`.
- Pandas-Profiling is now ydata-profiling; ensure the latest version is used.
- D-Tale requires a running server; use `open_browser=False` to avoid automatic browser opening.
- Check HTML reports for detailed insights (e.g., correlations, missing values, distributions).
- For large datasets, consider using minimal=True in pandas-profiling to reduce computation time.
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass