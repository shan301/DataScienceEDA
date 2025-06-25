# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:12:34 2025

@author: Shantanu
"""

"""Anomaly Detection
Anomaly detection identifies unusual or rare data points in a dataset, which may indicate errors, fraud, or significant events. This script covers statistical methods (z-scores, IQR), clustering-based methods (DBSCAN), and machine learning approaches (Isolation Forest) for detecting anomalies, using sample datasets (sales.csv, hr_data.csv, financials.csv).
"""

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import plotly.express as px
import matplotlib.pyplot as plt

"""1. Loading Sample Data
Load datasets for anomaly detection (sales.csv, hr_data.csv, financials.csv)."""
# Load datasets
sales_df = pd.read_csv('data/sales.csv', parse_dates=['Date'])
hr_df = pd.read_csv('data/hr_data.csv')
financials_df = pd.read_csv('data/financials.csv', parse_dates=['Date'])
print(sales_df.head())  # Output: First 5 rows of sales.csv
print(hr_df.head())     # Output: First 5 rows of hr_data.csv
print(financials_df.head())  # Output: First 5 rows of financials.csv

"""2. Z-Score Method for Anomaly Detection
Z-scores measure how many standard deviations a data point is from the mean."""
def detect_zscore_anomalies(df, column, threshold=3):
    """Detect anomalies using z-scores.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to analyze.
        threshold (float): Z-score threshold for anomalies.
    
    Returns:
        pd.DataFrame: DataFrame with anomaly flags.
    """
    df['Z_Score'] = (df[column] - df[column].mean()) / df[column].std()
    df['Z_Anomaly'] = df['Z_Score'].abs() > threshold
    return df

# Apply z-score to sales.csv
sales_zscore = detect_zscore_anomalies(sales_df.copy(), 'Amount', threshold=3)
print(sales_zscore[sales_df['Z_Anomaly']][['Amount', 'Z_Score']])  # Output: Rows flagged as anomalies with z-scores

"""3. Visualizing Z-Score Anomalies
Plot the data with anomalies highlighted."""
# Scatter plot for z-score anomalies
fig = px.scatter(sales_zscore, x='Date', y='Amount', color='Z_Anomaly',
                 title='Z-Score Anomalies in Sales Amount',
                 color_discrete_map={False: 'blue', True: 'red'})
fig.update_layout(xaxis_title='Date', yaxis_title='Amount')
fig.show()  # Output: Interactive scatter plot with anomalies in red

"""3. IQR Method for Anomaly Detection
The Interquartile Range (IQR) method identifies outliers based on quartiles."""
def detect_iqr_anomalies(df, column):
    """Detect anomalies using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to analyze.
    
    Returns:
        pd.DataFrame: DataFrame with anomaly flags.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['IQR_Anomaly'] = (df[column] < lower_bound) | (df[column] > upper_bound)
    return df

# Apply IQR to hr_data.csv
hr_iqr = detect_iqr_anomalies(hr_df.copy(), 'Salary')
print(hr_iqr[hr_iqr['IQR_Anomaly']][['Salary']])  # Output: Rows flagged as anomalies

"""4. Visualizing IQR Anomalies
Plot a box plot to visualize IQR-based anomalies."""
# Box plot for IQR anomalies
fig = px.box(hr_iqr, x='Dept', y='Salary', color='IQR_Anomaly',
             title='IQR Anomalies in Salary by Department',
             color_discrete_map={False: 'blue', True: 'red'})
fig.update_layout(xaxis_title='Department', yaxis_title='Salary')
fig.show()  # Output: Interactive box plot with anomalies in red

"""5. DBSCAN for Anomaly Detection
DBSCAN (Density-Based Spatial Clustering) identifies outliers as points not belonging to clusters."""
def detect_dbscan_anomalies(df, columns, eps=0.5, min_samples=5):
    """Detect anomalies using DBSCAN.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to analyze.
        eps (float): Maximum distance between two samples for clustering.
        min_samples (int): Minimum samples in a cluster.
    
    Returns:
        pd.DataFrame: DataFrame with anomaly flags.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[columns])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN_Anomaly'] = dbscan.fit_predict(X) == -1  # -1 indicates outliers
    return df

# Apply DBSCAN to hr_data.csv
hr_dbscan = detect_dbscan_anomalies(hr_df.copy(), ['Age', 'Salary', 'Experience'], eps=0.5, min_samples=5)
print(hr_dbscan[hr_dbscan['DBSCAN_Anomaly']][['Age', 'Salary', 'Experience']])  # Output: Rows flagged as anomalies

"""6. Visualizing DBSCAN Anomalies
Plot a 3D scatter plot to visualize DBSCAN anomalies."""
# 3D scatter plot for DBSCAN anomalies
fig = px.scatter_3d(hr_dbscan, x='Age', y='Salary', z='Experience', color='DBSCAN_Anomaly',
                    title='DBSCAN Anomalies in HR Data',
                    color_discrete_map={False: 'blue', True: 'red'})
fig.update_layout(scene=dict(xaxis_title='Age', yaxis_title='Salary', zaxis_title='Experience'))
fig.show()  # Output: Interactive 3D scatter plot with anomalies in red

"""7. Isolation Forest for Anomaly Detection
Isolation Forest isolates anomalies by randomly partitioning data."""
def detect_isolation_forest_anomalies(df, columns, contamination=0.1):
    """Detect anomalies using Isolation Forest.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to analyze.
        contamination (float): Expected proportion of anomalies.
    
    Returns:
        pd.DataFrame: DataFrame with anomaly flags.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[columns])
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['IsoForest_Anomaly'] = iso_forest.fit_predict(X) == -1  # -1 indicates anomalies
    return df

# Apply Isolation Forest to financials.csv
financials_iso = detect_isolation_forest_anomalies(financials_df.copy(), ['Revenue', 'Expenses'], contamination=0.1)
print(financials_iso[financials_iso['IsoForest_Anomaly']][['Revenue', 'Expenses']])  # Output: Rows flagged as anomalies

"""8. Visualizing Isolation Forest Anomalies
Plot a scatter plot to visualize Isolation Forest anomalies."""
# Scatter plot for Isolation Forest anomalies
fig = px.scatter(financials_iso, x='Revenue', y='Expenses', color='IsoForest_Anomaly',
                 title='Isolation Forest Anomalies in Financial Data',
                 color_discrete_map={False: 'blue', True: 'red'})
fig.update_layout(xaxis_title='Revenue', yaxis_title='Expenses')
fig.show()  # Output: Interactive scatter plot with anomalies in red

"""Exercises
Practice anomaly detection techniques with the following exercises using sales.csv, hr_data.csv, and financials.csv."""

"""Exercise 1: Load and Inspect Data
Load financials.csv and display the first 10 rows."""
financials_ex1 = pd.read_csv('data/financials.csv', parse_dates=['Date'])
print(financials_ex1.head(10))  # Output: First 10 rows of financials.csv

"""Exercise 2: Z-Score Anomalies
Detect anomalies in the Amount column of sales.csv using z-scores (threshold=2.5)."""
sales_ex2 = detect_zscore_anomalies(sales_df.copy(), 'Amount', threshold=2.5)
print(sales_ex2[sales_ex2['Z_Anomaly']][['Amount', 'Z_Score']])  # Output: Anomalous rows with z-scores

"""Exercise 3: Visualize Z-Score Anomalies
Create a scatter plot for sales.csv showing Amount vs Date with z-score anomalies highlighted."""
fig_ex3 = px.scatter(sales_ex2, x='Date', y='Amount', color='Z_Anomaly',
                     title='Z-Score Anomalies in Sales Amount',
                     color_discrete_map={False: 'blue', True: 'red'})
fig_ex3.update_layout(xaxis_title='Date', yaxis_title='Amount')
fig_ex3.show()  # Output: Scatter plot with anomalies in red

"""Exercise 4: IQR Anomalies
Detect anomalies in the Salary column of hr_data.csv using the IQR method."""
hr_ex4 = detect_iqr_anomalies(hr_df.copy(), 'Salary')
print(hr_ex4[hr_ex4['IQR_Anomaly']][['Salary']])  # Output: Anomalous rows

"""Exercise 5: Visualize IQR Anomalies
Create a box plot for hr_data.csv showing Salary by Department with IQR anomalies highlighted."""
fig_ex5 = px.box(hr_ex4, x='Dept', y='Salary', color='IQR_Anomaly',
                 title='IQR Anomalies in Salary by Department',
                 color_discrete_map={False: 'blue', True: 'red'})
fig_ex5.update_layout(xaxis_title='Department', yaxis_title='Salary')
fig_ex5.show()  # Output: Box plot with anomalies in red

"""Exercise 6: DBSCAN Anomalies
Detect anomalies in hr_data.csv using DBSCAN on Age and Salary (eps=0.6, min_samples=4)."""
hr_ex6 = detect_dbscan_anomalies(hr_df.copy(), ['Age', 'Salary'], eps=0.6, min_samples=4)
print(hr_ex6[hr_ex6['DBSCAN_Anomaly']][['Age', 'Salary']])  # Output: Anomalous rows

"""Exercise 7: Visualize DBSCAN Anomalies
Create a scatter plot for hr_data.csv showing Age vs Salary with DBSCAN anomalies highlighted."""
fig_ex7 = px.scatter(hr_ex6, x='Age', y='Salary', color='DBSCAN_Anomaly',
                     title='DBSCAN Anomalies in HR Data',
                     color_discrete_map={False: 'blue', True: 'red'})
fig_ex7.update_layout(xaxis_title='Age', yaxis_title='Salary')
fig_ex7.show()  # Output: Scatter plot with anomalies in red

"""Exercise 8: Isolation Forest Anomalies
Detect anomalies in sales.csv using Isolation Forest on Amount and ProductID (contamination=0.05)."""
sales_ex8 = detect_isolation_forest_anomalies(sales_df.copy(), ['Amount', 'ProductID'], contamination=0.05)
print(sales_ex8[sales_ex8['IsoForest_Anomaly']][['Amount', 'ProductID']])  # Output: Anomalous rows

"""Exercise 9: Visualize Isolation Forest Anomalies
Create a scatter plot for sales.csv showing Amount vs ProductID with Isolation Forest anomalies highlighted."""
fig_ex9 = px.scatter(sales_ex8, x='ProductID', y='Amount', color='IsoForest_Anomaly',
                     title='Isolation Forest Anomalies in Sales Data',
                     color_discrete_map={False: 'blue', True: 'red'})
fig_ex9.update_layout(xaxis_title='ProductID', yaxis_title='Amount')
fig_ex9.show()  # Output: Scatter plot with anomalies in red

"""Exercise 10: Compare Z-Score and IQR
Detect anomalies in financials.csv (Revenue) using both z-score (threshold=3) and IQR, and count overlaps."""
financials_ex10 = detect_zscore_anomalies(financials_df.copy(), 'Revenue', threshold=3)
financials_ex10 = detect_iqr_anomalies(financials_ex10, 'Revenue')
overlap_count = financials_ex10[financials_ex10['Z_Anomaly'] & financials_ex10['IQR_Anomaly']].shape[0]
print(f"Number of overlapping anomalies: {overlap_count}")  # Output: Count of overlapping anomalies

"""Exercise 11: Time Series Anomalies with Z-Score
Detect anomalies in time_series.csv (Value) using z-scores and visualize."""
time_series_df = pd.read_csv('data/time_series.csv', parse_dates=['Date'])
time_series_ex11 = detect_zscore_anomalies(time_series_df.copy(), 'Value', threshold=3)
fig_ex11 = px.scatter(time_series_ex11, x='Date', y='Value', color='Z_Anomaly',
                      title='Z-Score Anomalies in Time Series',
                      color_discrete_map={False: 'blue', True: 'red'})
fig_ex11.update_layout(xaxis_title='Date', yaxis_title='Value')
fig_ex11.show()  # Output: Scatter plot with anomalies in red

"""Exercise 12: DBSCAN with Custom Parameters
Detect anomalies in financials.csv using DBSCAN on Revenue and Expenses (eps=0.7, min_samples=3)."""
financials_ex12 = detect_dbscan_anomalies(financials_df.copy(), ['Revenue', 'Expenses'], eps=0.7, min_samples=3)
print(financials_ex12[financials_ex12['DBSCAN_Anomaly']][['Revenue', 'Expenses']])  # Output: Anomalous rows

"""Exercise 13: Isolation Forest with Adjusted Contamination
Detect anomalies in hr_data.csv using Isolation Forest on Age, Salary, Experience (contamination=0.15)."""
hr_ex13 = detect_isolation_forest_anomalies(hr_df.copy(), ['Age', 'Salary', 'Experience'], contamination=0.15)
print(hr_ex13[hr_ex13['IsoForest_Anomaly']][['Age', 'Salary', 'Experience']])  # Output: Anomalous rows

"""Exercise 14: Anomaly Proportion
Calculate the proportion of anomalies detected by IQR in sales.csv (Amount)."""
sales_ex14 = detect_iqr_anomalies(sales_df.copy(), 'Amount')
anomaly_proportion = sales_ex14['IQR_Anomaly'].mean()
print(f"Proportion of IQR anomalies: {anomaly_proportion:.4f}")  # Output: Proportion of anomalies (e.g., 0.0500)

"""Exercise 15: Visualize Multiple Methods
Create a scatter plot for financials.csv (Revenue vs Expenses) showing anomalies from z-score and Isolation Forest."""
financials_ex15 = detect_zscore_anomalies(financials_df.copy(), 'Revenue', threshold=3)
financials_ex15 = detect_isolation_forest_anomalies(financials_ex15, ['Revenue', 'Expenses'], contamination=0.1)
financials_ex15['Anomaly_Type'] = 'None'
financials_ex15.loc[financials_ex15['Z_Anomaly'], 'Anomaly_Type'] = 'Z-Score'
financials_ex15.loc[financials_ex15['IsoForest_Anomaly'], 'Anomaly_Type'] = 'Isolation Forest'
financials_ex15.loc[financials_ex15['Z_Anomaly'] & financials_ex15['IsoForest_Anomaly'], 'Anomaly_Type'] = 'Both'
fig_ex15 = px.scatter(financials_ex15, x='Revenue', y='Expenses', color='Anomaly_Type',
                      title='Z-Score vs Isolation Forest Anomalies',
                      color_discrete_map={'None': 'blue', 'Z-Score': 'red', 'Isolation Forest': 'green', 'Both': 'purple'})
fig_ex15.update_layout(xaxis_title='Revenue', yaxis_title='Expenses')
fig_ex15.show()  # Output: Scatter plot with anomalies by method

"""Notes
- Ensure datasets (sales.csv, hr_data.csv, financials.csv) have appropriate columns (e.g., Date, Amount, Region, Age, Salary, Experience, Revenue, Expenses).
- Adjust thresholds (z-score, IQR), eps/min_samples (DBSCAN), and contamination (Isolation Forest) based on dataset characteristics.
- Install required libraries: `pip install pandas numpy scikit-learn plotly`.
- For advanced anomaly detection, consider autoencoders or time-series-specific methods (not covered here).
- Visualizations use Plotly for interactivity; ensure a compatible environment (e.g., Jupyter or browser).
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass