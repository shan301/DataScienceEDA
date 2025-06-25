# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:53:19 2025

@author: Shantanu
"""

"""Time Series Analysis
Time series analysis involves studying data points collected over time to identify patterns such as trends, seasonality, and cycles. This script covers essential techniques for analyzing time series data, including visualization, decomposition, rolling statistics, and autocorrelation, using a sample dataset (time_series.csv).
"""

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

"""1. Loading and Preparing Time Series Data
Time series data requires a datetime index for proper analysis. We load the data and ensure the date column is parsed correctly."""
# Load time series data
df = pd.read_csv('data/time_series.csv', parse_dates=['Date'], index_col='Date')
print(df.head())  # Output: First 5 rows with Date as index and Value column
print(df.info())  # Output: Data types and non-null counts

"""2. Visualizing Time Series Data
Plotting the time series helps identify trends and seasonality visually."""
# Plot time series
plt.figure(figsize=(10, 5))
plt.plot(df['Value'], label='Value')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()  # Output: Line plot of Value over time

"""3. Trend Analysis with Rolling Statistics
Rolling statistics (e.g., moving average) smooth the data to highlight trends."""
# Calculate 7-day rolling mean
df['Rolling_Mean'] = df['Value'].rolling(window=7).mean()
df['Rolling_Std'] = df['Value'].rolling(window=7).std()

# Plot with rolling statistics
plt.figure(figsize=(10, 5))
plt.plot(df['Value'], label='Value')
plt.plot(df['Rolling_Mean'], label='7-Day Rolling Mean', color='red')
plt.plot(df['Rolling_Std'], label='7-Day Rolling Std', color='green')
plt.title('Time Series with Rolling Statistics')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()  # Output: Plot with original data, rolling mean, and rolling std

"""4. Seasonal Decomposition
Decomposition separates a time series into trend, seasonal, and residual components."""
# Decompose time series (assuming monthly seasonality)
result = seasonal_decompose(df['Value'], model='additive', period=30)
result.plot()
plt.show()  # Output: Four subplots (observed, trend, seasonal, residual)

"""5. Autocorrelation Analysis
Autocorrelation measures how a time series correlates with its lagged versions, useful for identifying seasonality."""
# Plot autocorrelation
plot_acf(df['Value'], lags=50)
plt.title('Autocorrelation Plot')
plt.show()  # Output: ACF plot showing correlation at different lags

"""6. Stationarity Check
Stationarity is crucial for many time series models. We use the Augmented Dickey-Fuller (ADF) test to check for stationarity."""
# Perform ADF test
adf_result = adfuller(df['Value'])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')  # Output: ADF statistic and p-value
# If p-value < 0.05, the series is stationary

"""7. Differencing for Stationarity
Differencing removes trends to make the series stationary."""
# Apply first-order differencing
df['Diff_Value'] = df['Value'].diff()
df['Diff_Value'].plot(title='Differenced Time Series')
plt.show()  # Output: Plot of differenced series

# Check stationarity of differenced series
adf_diff = adfuller(df['Diff_Value'].dropna())
print(f'ADF Statistic (Diff): {adf_diff[0]}')
print(f'p-value (Diff): {adf_diff[1]}')  # Output: ADF statistic and p-value for differenced series

"""8. Simple Moving Average Forecasting
A simple moving average can be used for basic forecasting."""
# Forecast using 7-day moving average
df['Forecast'] = df['Value'].rolling(window=7).mean().shift(1)

# Plot actual vs forecast
plt.figure(figsize=(10, 5))
plt.plot(df['Value'], label='Actual')
plt.plot(df['Forecast'], label='Moving Average Forecast', color='orange')
plt.title('Moving Average Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()  # Output: Plot comparing actual and forecasted values

"""Exercises
Practice the concepts learned with the following exercises using time_series.csv."""

"""Exercise 1: Load and Plot
Load time_series.csv and create a line plot of the Value column."""
df_ex1 = pd.read_csv('data/time_series.csv', parse_dates=['Date'], index_col='Date')
plt.plot(df_ex1['Value'])
plt.title('Time Series Plot')
plt.show()  # Output: Line plot of Value

"""Exercise 2: Rolling Mean
Calculate and plot a 14-day rolling mean for the Value column."""
df['Rolling_Mean_14'] = df['Value'].rolling(window=14).mean()
plt.plot(df['Value'], label='Value')
plt.plot(df['Rolling_Mean_14'], label='14-Day Rolling Mean', color='red')
plt.legend()
plt.show()  # Output: Plot with Value and 14-day rolling mean

"""Exercise 3: Seasonal Decomposition
Perform seasonal decomposition with a period of 7 days and plot the components."""
result_ex3 = seasonal_decompose(df['Value'], model='additive', period=7)
result_ex3.plot()
plt.show()  # Output: Four subplots (observed, trend, seasonal, residual)

"""Exercise 4: Autocorrelation
Plot the autocorrelation of Value for 30 lags."""
plot_acf(df['Value'], lags=30)
plt.title('Autocorrelation Plot')
plt.show()  # Output: ACF plot for 30 lags

"""Exercise 5: Stationarity Test
Perform an ADF test on Value and interpret the results."""
adf_result_ex5 = adfuller(df['Value'])
print(f'ADF Statistic: {adf_result_ex5[0]}')
print(f'p-value: {adf_result_ex5[1]}')  # Output: ADF statistic and p-value
# Interpretation: If p-value < 0.05, series is stationary

"""Exercise 6: Differencing
Apply second-order differencing and plot the result."""
df['Diff2_Value'] = df['Value'].diff().diff()
df['Diff2_Value'].plot(title='Second-Order Differenced Time Series')
plt.show()  # Output: Plot of second-order differenced series

"""Exercise 7: Moving Average Forecast
Create a 30-day moving average forecast and plot it against actual values."""
df['Forecast_30'] = df['Value'].rolling(window=30).mean().shift(1)
plt.plot(df['Value'], label='Actual')
plt.plot(df['Forecast_30'], label='30-Day MA Forecast', color='orange')
plt.legend()
plt.show()  # Output: Plot comparing actual and 30-day forecast

"""Exercise 8: Rolling Standard Deviation
Calculate and plot a 7-day rolling standard deviation."""
df['Rolling_Std_7'] = df['Value'].rolling(window=7).std()
plt.plot(df['Rolling_Std_7'], label='7-Day Rolling Std', color='green')
plt.title('Rolling Standard Deviation')
plt.legend()
plt.show()  # Output: Plot of 7-day rolling standard deviation

"""Notes
- Ensure time_series.csv has a Date column in a parseable format (e.g., YYYY-MM-DD) and a numeric Value column.
- Adjust the period parameter in seasonal_decompose based on the data’s frequency (e.g., 7 for daily data with weekly seasonality, 30 for monthly).
- For advanced forecasting, consider models like ARIMA or Prophet (not covered here).
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass