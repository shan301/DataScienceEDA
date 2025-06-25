# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 16:05:46 2025

@author: Shantanu
"""

"""Categorical Data Analysis
Categorical data analysis involves exploring and summarizing non-numerical data, such as categories or labels, using frequency tables, bar plots, and other techniques. This script demonstrates key concepts for analyzing categorical variables using Python, pandas, and visualization libraries.

1. Loading and Inspecting Categorical Data
Load a dataset and identify categorical columns using pandas."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_inspect_data(file_path):
    df = pd.read_csv(file_path)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"Categorical columns: {list(categorical_cols)}")
    return df, categorical_cols

# Example usage
df, categorical_cols = load_and_inspect_data('data/customers.csv')
print(df[categorical_cols].head())

"""2. Frequency Tables
Generate frequency tables to summarize the distribution of categorical variables."""
def frequency_table(column):
    freq = column.value_counts()
    print(f"Frequency table for {column.name}:\n{freq}")
    return freq

# Example usage
freq_table(df['Gender'])

"""3. Relative Frequency
Calculate relative frequencies (proportions) for a categorical column."""
def relative_frequency(column):
    rel_freq = column.value_counts(normalize=True) * 100
    print(f"Relative frequency (%) for {column.name}:\n{rel_freq}")
    return rel_freq

# Example usage
relative_frequency(df['Gender'])

"""4. Bar Plots
Visualize categorical data using bar plots with matplotlib or seaborn."""
def plot_bar(column, title="Bar Plot"):
    plt.figure(figsize=(8, 6))
    column.value_counts().plot(kind='bar')
    plt.title(title)
    plt.xlabel(column.name)
    plt.ylabel('Count')
    plt.show()

# Example usage
plot_bar(df['Gender'], title="Gender Distribution")

"""5. Stacked Bar Plots
Create stacked bar plots to compare two categorical variables."""
def stacked_bar_plot(df, col1, col2):
    crosstab = pd.crosstab(df[col1], df[col2])
    crosstab.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title(f"Stacked Bar Plot of {col1} by {col2}")
    plt.xlabel(col1)
    plt.ylabel('Count')
    plt.show()

# Example usage
stacked_bar_plot(df, 'Gender', 'Region')

"""6. Contingency Tables
Create contingency tables to analyze the relationship between two categorical variables."""
def contingency_table(df, col1, col2):
    crosstab = pd.crosstab(df[col1], df[col2], margins=True)
    print(f"Contingency table for {col1} and {col2}:\n{crosstab}")
    return crosstab

# Example usage
contingency_table(df, 'Gender', 'Region')

"""7. Chi-Square Test
Perform a chi-square test to check for independence between two categorical variables."""
from scipy.stats import chi2_contingency

def chi_square_test(df, col1, col2):
    crosstab = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    print(f"Chi-Square Test: chi2={chi2:.4f}, p-value={p:.4f}")
    return chi2, p

# Example usage
chi_square_test(df, 'Gender', 'Region')

"""8. Pie Charts
Visualize the proportion of categories using pie charts."""
def plot_pie(column, title="Pie Chart"):
    plt.figure(figsize=(8, 6))
    column.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(title)
    plt.ylabel('')
    plt.show()

# Example usage
plot_pie(df['Gender'], title="Gender Proportion")

"""9. Encoding Categorical Data
Convert categorical data into numerical form using label encoding or one-hot encoding."""
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df, column):
    le = LabelEncoder()
    df[f'{column}_encoded'] = le.fit_transform(df[column])
    print(f"Encoded {column}:\n{df[[column, f'{column}_encoded']].head()}")
    return df

# Example usage
df = encode_categorical(df, 'Gender')

def one_hot_encode(df, column):
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, one_hot], axis=1)
    print(f"One-hot encoded {column}:\n{df[[column] + list(one_hot.columns)].head()}")
    return df

# Example usage
df = one_hot_encode(df, 'Region')

"""10. Mode for Imputation
Use the mode to impute missing values in categorical columns."""
def impute_mode(df, column):
    mode_value = df[column].mode()[0]
    df[column].fillna(mode_value, inplace=True)
    print(f"Imputed missing values in {column} with mode: {mode_value}")
    return df

# Example usage
df = impute_mode(df, 'Gender')

"""11. Grouped Analysis
Analyze numerical variables grouped by categorical variables."""
def grouped_analysis(df, cat_col, num_col):
    grouped = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'count'])
    print(f"Grouped analysis of {num_col} by {cat_col}:\n{grouped}")
    return grouped

# Example usage
grouped_analysis(df, 'Gender', 'Salary')

"""12. Mosaic Plots
Create mosaic plots to visualize relationships between categorical variables."""
from statsmodels.graphics.mosaicplot import mosaic

def mosaic_plot(df, col1, col2):
    plt.figure(figsize=(8, 6))
    mosaic(df, [col1, col2])
    plt.title(f"Mosaic Plot of {col1} and {col2}")
    plt.show()

# Example usage
mosaic_plot(df, 'Gender', 'Region')

"""13. Handling High Cardinality
Reduce high-cardinality categorical variables by grouping rare categories."""
def reduce_cardinality(df, column, threshold=0.05):
    freq = df[column].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    df[f'{column}_grouped'] = df[column].apply(lambda x: 'Other' if x in rare else x)
    print(f"Reduced cardinality for {column}:\n{df[f'{column}_grouped'].value_counts()}")
    return df

# Example usage
df = reduce_cardinality(df, 'Region', threshold=0.05)

"""14. Interactive Visualizations
Create interactive bar plots using plotly."""
import plotly.express as px

def interactive_bar(df, column):
    fig = px.histogram(df, x=column, title=f"Interactive Bar Plot of {column}")
    fig.show()

# Example usage
interactive_bar(df, 'Gender')

"""15. Custom Aggregation
Perform custom aggregation on categorical data for advanced insights."""
def custom_aggregation(df, cat_col, num_col):
    agg = df.groupby(cat_col)[num_col].apply(lambda x: x.max() - x.min())
    print(f"Custom aggregation (range) of {num_col} by {cat_col}:\n{agg}")
    return agg

# Example usage
custom_aggregation(df, 'Gender', 'Salary')

"""Exercises for Categorical Data Analysis
Exercise 1: Frequency Table
Write a function to print a frequency table for a given categorical column."""
def ex_frequency_table(df, column):
    freq = df[column].value_counts()
    print(f"Frequency table for {column}:\n{freq}")
    return freq

# Example usage
ex_frequency_table(df, 'Region')

"""Exercise 2: Relative Frequency
Write a function to calculate relative frequencies for a categorical column."""
def ex_relative_frequency(df, column):
    rel_freq = df[column].value_counts(normalize=True) * 100
    print(f"Relative frequency (%) for {column}:\n{rel_freq}")
    return rel_freq

# Example usage
ex_relative_frequency(df, 'Region')

"""Exercise 3: Bar Plot
Write a function to create a bar plot for a categorical column."""
def ex_bar_plot(df, column):
    plt.figure(figsize=(8, 6))
    df[column].value_counts().plot(kind='bar')
    plt.title(f"Bar Plot of {column}")
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

# Example usage
ex_bar_plot(df, 'Region')

"""Exercise 4: Pie Chart
Write a function to create a pie chart for a categorical column."""
def ex_pie_chart(df, column):
    plt.figure(figsize=(8, 6))
    df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f"Pie Chart of {column}")
    plt.ylabel('')
    plt.show()

# Example usage
ex_pie_chart(df, 'Region')

"""Exercise 5: Contingency Table
Write a function to create a contingency table for two categorical columns."""
def ex_contingency_table(df, col1, col2):
    crosstab = pd.crosstab(df[col1], df[col2], margins=True)
    print(f"Contingency table for {col1} and {col2}:\n{crosstab}")
    return crosstab

# Example usage
ex_contingency_table(df, 'Gender', 'Region')

"""Exercise 6: Chi-Square Test
Write a function to perform a chi-square test for two categorical columns."""
def ex_chi_square_test(df, col1, col2):
    crosstab = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    print(f"Chi-Square Test: chi2={chi2:.4f}, p-value={p:.4f}")
    return chi2, p

# Example usage
ex_chi_square_test(df, 'Gender', 'Region')

"""Exercise 7: Label Encoding
Write a function to perform label encoding on a categorical column."""
def ex_label_encode(df, column):
    le = LabelEncoder()
    df[f'{column}_encoded'] = le.fit_transform(df[column])
    print(f"Encoded {column}:\n{df[[column, f'{column}_encoded']].head()}")
    return df

# Example usage
df = ex_label_encode(df, 'Region')

"""Exercise 8: One-Hot Encoding
Write a function to perform one-hot encoding on a categorical column."""
def ex_one_hot_encode(df, column):
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, one_hot], axis=1)
    print(f"One-hot encoded {column}:\n{df[[column] + list(one_hot.columns)].head()}")
    return df

# Example usage
df = ex_one_hot_encode(df, 'Gender')

"""Exercise 9: Mode Imputation
Write a function to impute missing values in a categorical column using the mode."""
def ex_impute_mode(df, column):
    mode_value = df[column].mode()[0]
    df[column].fillna(mode_value, inplace=True)
    print(f"Imputed missing values in {column} with mode: {mode_value}")
    return df

# Example usage
df = ex_impute_mode(df, 'Region')

"""Exercise 10: Grouped Analysis
Write a function to perform grouped analysis of a numerical column by a categorical column."""
def ex_grouped_analysis(df, cat_col, num_col):
    grouped = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'count'])
    print(f"Grouped analysis of {num_col} by {cat_col}:\n{grouped}")
    return grouped

# Example usage
ex_grouped_analysis(df, 'Region', 'Salary')

"""Exercise 11: Stacked Bar Plot
Write a function to create a stacked bar plot for two categorical columns."""
def ex_stacked_bar_plot(df, col1, col2):
    crosstab = pd.crosstab(df[col1], df[col2])
    crosstab.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title(f"Stacked Bar Plot of {col1} by {col2}")
    plt.xlabel(col1)
    plt.ylabel('Count')
    plt.show()

# Example usage
ex_stacked_bar_plot(df, 'Gender', 'Region')

"""Exercise 12: Mosaic Plot
Write a function to create a mosaic plot for two categorical columns."""
def ex_mosaic_plot(df, col1, col2):
    plt.figure(figsize=(8, 6))
    mosaic(df, [col1, col2])
    plt.title(f"Mosaic Plot of {col1} and {col2}")
    plt.show()

# Example usage
ex_mosaic_plot(df, 'Gender', 'Region')

"""Exercise 13: Reduce Cardinality
Write a function to reduce high cardinality in a categorical column."""
def ex_reduce_cardinality(df, column, threshold=0.05):
    freq = df[column].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    df[f'{column}_grouped'] = df[column].apply(lambda x: 'Other' if x in rare else x)
    print(f"Reduced cardinality for {column}:\n{df[f'{column}_grouped'].value_counts()}")
    return df

# Example usage
df = ex_reduce_cardinality(df, 'Region', threshold=0.05)

"""Exercise 14: Interactive Bar Plot
Write a function to create an interactive bar plot using plotly."""
def ex_interactive_bar(df, column):
    fig = px.histogram(df, x=column, title=f"Interactive Bar Plot of {column}")
    fig.show()

# Example usage
ex_interactive_bar(df, 'Region')

"""Exercise 15: Custom Aggregation
Write a function to perform custom aggregation on a numerical column by a categorical column."""
def ex_custom_aggregation(df, cat_col, num_col):
    agg = df.groupby(cat_col)[num_col].apply(lambda x: x.max() - x.min())
    print(f"Custom aggregation (range) of {num_col} by {cat_col}:\n{agg}")
    return agg

# Example usage
ex_custom_aggregation(df, 'Region', 'Salary')