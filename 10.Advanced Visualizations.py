# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:08:33 2025

@author: Shantanu
"""

"""Advanced Visualizations
Advanced visualizations enhance data exploration with interactive plots, 3D visualizations, and animations. This script covers techniques using Plotly for interactive plots, 3D scatter plots, and animated visualizations, applied to sample datasets (sales.csv, hr_data.csv, time_series.csv).
"""

# Import required libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

"""1. Loading Sample Data
Load datasets for visualization (sales.csv, hr_data.csv, time_series.csv)."""
# Load datasets
sales_df = pd.read_csv('data/sales.csv', parse_dates=['Date'])
hr_df = pd.read_csv('data/hr_data.csv')
time_series_df = pd.read_csv('data/time_series.csv', parse_dates=['Date'])
print(sales_df.head())  # Output: First 5 rows of sales.csv
print(hr_df.head())    # Output: First 5 rows of hr_data.csv
print(time_series_df.head())  # Output: First 5 rows of time_series.csv

"""2. Interactive Line Plot with Plotly
Create an interactive line plot for time series data."""
# Interactive line plot for time_series.csv
fig = px.line(time_series_df, x='Date', y='Value', title='Interactive Time Series Plot')
fig.update_layout(xaxis_title='Date', yaxis_title='Value')
fig.show()  # Output: Interactive line plot with hover, zoom, and pan

"""3. Interactive Scatter Plot
Visualize relationships between variables with an interactive scatter plot."""
# Scatter plot for hr_data.csv (Age vs Salary, colored by Department)
fig = px.scatter(hr_df, x='Age', y='Salary', color='Dept', size='Experience',
                 title='Age vs Salary by Department', hover_data=['EmpID'])
fig.update_layout(xaxis_title='Age', yaxis_title='Salary')
fig.show()  # Output: Interactive scatter plot with color and size encoding

"""4. 3D Scatter Plot
Create a 3D scatter plot to visualize three variables simultaneously."""
# 3D scatter plot for hr_data.csv (Age, Salary, Experience)
fig = px.scatter_3d(hr_df, x='Age', y='Salary', z='Experience', color='Dept',
                    title='3D Scatter Plot: Age, Salary, Experience',
                    hover_data=['EmpID'])
fig.update_layout(scene=dict(xaxis_title='Age', yaxis_title='Salary', zaxis_title='Experience'))
fig.show()  # Output: Interactive 3D scatter plot

"""5. Heatmap with Plotly
Visualize correlations or aggregated data with an interactive heatmap."""
# Correlation heatmap for hr_data.csv
corr_matrix = hr_df[['Age', 'Salary', 'Experience']].corr()
fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Heatmap',
                labels=dict(color='Correlation'))
fig.update_layout(xaxis_title='Features', yaxis_title='Features')
fig.show()  # Output: Interactive heatmap with correlation values

"""6. Animated Line Plot
Create an animated line plot to show changes over time."""
# Animated line plot for time_series.csv
fig = px.line(time_series_df, x='Date', y='Value', animation_frame=time_series_df['Date'].dt.year,
              title='Animated Time Series Plot')
fig.update_layout(xaxis_title='Date', yaxis_title='Value')
fig.show()  # Output: Animated line plot with year-based slider

"""7. Box Plot with Plotly
Visualize distributions with an interactive box plot."""
# Box plot for sales.csv (Amount by Region)
fig = px.box(sales_df, x='Region', y='Amount', title='Sales Amount by Region',
             color='Region')
fig.update_layout(xaxis_title='Region', yaxis_title='Amount')
fig.show()  # Output: Interactive box plot with hover details

"""8. Parallel Coordinates Plot
Visualize multivariate data with a parallel coordinates plot."""
# Parallel coordinates for hr_data.csv
fig = px.parallel_coordinates(hr_df, dimensions=['Age', 'Salary', 'Experience'],
                              color='Salary', title='Parallel Coordinates: HR Data')
fig.show()  # Output: Interactive parallel coordinates plot

"""Exercises
Practice advanced visualization techniques with the following exercises using sales.csv, hr_data.csv, and time_series.csv."""

"""Exercise 1: Load and Inspect Data
Load sales.csv and display the first 10 rows."""
sales_ex1 = pd.read_csv('data/sales.csv', parse_dates=['Date'])
print(sales_ex1.head(10))  # Output: First 10 rows of sales.csv

"""Exercise 2: Interactive Line Plot
Create an interactive line plot for sales.csv showing Amount over Date."""
fig_ex2 = px.line(sales_ex1, x='Date', y='Amount', title='Sales Amount Over Time')
fig_ex2.update_layout(xaxis_title='Date', yaxis_title='Amount')
fig_ex2.show()  # Output: Interactive line plot of sales amount

"""Exercise 3: Scatter Plot with Custom Hover
Create a scatter plot for hr_data.csv (Salary vs Experience, colored by Age) with EmpID in hover."""
fig_ex3 = px.scatter(hr_df, x='Salary', y='Experience', color='Age', hover_data=['EmpID'],
                     title='Salary vs Experience by Age')
fig_ex3.update_layout(xaxis_title='Salary', yaxis_title='Experience')
fig_ex3.show()  # Output: Interactive scatter plot with custom hover

"""Exercise 4: 3D Scatter Plot
Create a 3D scatter plot for hr_data.csv (Age, Salary, Experience, colored by Salary)."""
fig_ex4 = px.scatter_3d(hr_df, x='Age', y='Salary', z='Experience', color='Salary',
                        title='3D Scatter: Age, Salary, Experience',
                        hover_data=['EmpID'])
fig_ex4.update_layout(scene=dict(xaxis_title='Age', yaxis_title='Salary', zaxis_title='Experience'))
fig_ex4.show()  # Output: Interactive 3D scatter plot

"""Exercise 5: Heatmap of Aggregated Data
Create a heatmap showing average Amount by Region and Month from sales.csv."""
sales_ex5 = sales_ex1.copy()
sales_ex5['Month'] = sales_ex5['Date'].dt.month
pivot_table = sales_ex5.pivot_table(values='Amount', index='Region', columns='Month', aggfunc='mean')
fig_ex5 = px.imshow(pivot_table, text_auto=True, title='Average Sales Amount by Region and Month')
fig_ex5.update_layout(xaxis_title='Month', yaxis_title='Region')
fig_ex5.show()  # Output: Interactive heatmap of average sales

"""Exercise 6: Animated Scatter Plot
Create an animated scatter plot for sales.csv (Amount vs Date, animated by year)."""
fig_ex6 = px.scatter(sales_ex1, x='Date', y='Amount', animation_frame=sales_ex1['Date'].dt.year,
                     title='Animated Sales Amount Over Time')
fig_ex6.update_layout(xaxis_title='Date', yaxis_title='Amount')
fig_ex6.show()  # Output: Animated scatter plot with year slider

"""Exercise 7: Box Plot by Department
Create a box plot for hr_data.csv showing Salary distribution by Department."""
fig_ex7 = px.box(hr_df, x='Dept', y='Salary', title='Salary Distribution by Department',
                 color='Dept')
fig_ex7.update_layout(xaxis_title='Department', yaxis_title='Salary')
fig_ex7.show()  # Output: Interactive box plot of salaries

"""Exercise 8: Violin Plot
Create a violin plot for hr_data.csv showing Salary distribution by Department."""
fig_ex8 = px.violin(hr_df, x='Dept', y='Salary', title='Salary Distribution by Department (Violin)',
                    color='Dept')
fig_ex8.update_layout(xaxis_title='Department', yaxis_title='Salary')
fig_ex8.show()  # Output: Interactive violin plot of salaries

"""Exercise 9: Parallel Coordinates
Create a parallel coordinates plot for sales.csv (Amount, ProductID, colored by Region)."""
fig_ex9 = px.parallel_coordinates(sales_ex1, dimensions=['Amount', 'ProductID'],
                                  color='Region', title='Parallel Coordinates: Sales Data')
fig_ex9.show()  # Output: Interactive parallel coordinates plot

"""Exercise 10: 3D Line Plot
Create a 3D line plot for time_series.csv (Date, Value, and a constant z-axis)."""
time_series_ex10 = time_series_df.copy()
time_series_ex10['Z'] = 0  # Constant z-axis
fig_ex10 = px.line_3d(time_series_ex10, x='Date', y='Value', z='Z',
                      title='3D Line Plot: Time Series')
fig_ex10.update_layout(scene=dict(xaxis_title='Date', yaxis_title='Value', zaxis_title='Z'))
fig_ex10.show()  # Output: Interactive 3D line plot

"""Exercise 11: Bubble Chart
Create a bubble chart for hr_data.csv (Age vs Salary, size by Experience, colored by Dept)."""
fig_ex11 = px.scatter(hr_df, x='Age', y='Salary', size='Experience', color='Dept',
                      title='Bubble Chart: Age vs Salary', hover_data=['EmpID'])
fig_ex11.update_layout(xaxis_title='Age', yaxis_title='Salary')
fig_ex11.show()  # Output: Interactive bubble chart

"""Exercise 12: Animated Bubble Chart
Create an animated bubble chart for sales.csv (Amount vs ProductID, animated by year)."""
fig_ex12 = px.scatter(sales_ex1, x='ProductID', y='Amount', animation_frame=sales_ex1['Date'].dt.year,
                      color='Region', size='Amount', title='Animated Bubble Chart: Sales Data')
fig_ex12.update_layout(xaxis_title='ProductID', yaxis_title='Amount')
fig_ex12.show()  # Output: Animated bubble chart with year slider

"""Exercise 13: Faceted Scatter Plot
Create a faceted scatter plot for hr_data.csv (Salary vs Experience, faceted by Department)."""
fig_ex13 = px.scatter(hr_df, x='Salary', y='Experience', facet_col='Dept',
                      title='Salary vs Experience by Department')
fig_ex13.update_layout(yaxis_title='Experience')
fig_ex13.show()  # Output: Faceted scatter plot by department

"""Exercise 14: Custom Heatmap
Create a heatmap for sales.csv showing count of sales by Region and ProductID."""
pivot_table_ex14 = sales_ex1.pivot_table(values='Amount', index='Region', columns='ProductID', aggfunc='count')
fig_ex14 = px.imshow(pivot_table_ex14, text_auto=True, title='Sales Count by Region and ProductID')
fig_ex14.update_layout(xaxis_title='ProductID', yaxis_title='Region')
fig_ex14.show()  # Output: Interactive heatmap of sales counts

"""Exercise 15: Combined Plot
Create a combined Plotly figure with a line plot and scatter plot for time_series.csv."""
fig_ex15 = go.Figure()
fig_ex15.add_trace(go.Scatter(x=time_series_df['Date'], y=time_series_df['Value'], mode='markers', name='Data Points'))
fig_ex15.add_trace(go.Scatter(x=time_series_df['Date'], y=time_series_df['Value'].rolling(window=7).mean(),
                              mode='lines', name='7-Day Moving Average', line=dict(color='red')))
fig_ex15.update_layout(title='Combined Line and Scatter Plot', xaxis_title='Date', yaxis_title='Value')
fig_ex15.show()  # Output: Combined interactive line and scatter plot

"""Notes
- Ensure datasets (sales.csv, hr_data.csv, time_series.csv) have appropriate columns (e.g., Date, Amount, Region, Age, Salary, Experience, Value).
- Plotly plots are interactive and require a browser or Jupyter environment to render.
- Install Plotly with `pip install plotly`.
- Adjust figure sizes and parameters (e.g., max_features, animation_frame) based on dataset size.
- For advanced visualizations, consider Plotly Dash for web-based dashboards (not covered here).
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass