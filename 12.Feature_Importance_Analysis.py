# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:15:41 2025

@author: Shantanu
"""

"""Feature Importance Analysis
Feature importance analysis identifies which features contribute most to a predictive model's performance. This script covers tree-based methods (Random Forest) and permutation importance for assessing feature importance, using sample datasets (sales.csv, hr_data.csv, customers.csv).
"""

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import plotly.express as px
import matplotlib.pyplot as plt

"""1. Loading Sample Data
Load datasets for feature importance analysis (sales.csv, hr_data.csv, customers.csv)."""
# Load datasets
sales_df = pd.read_csv('data/sales.csv', parse_dates=['Date'])
hr_df = pd.read_csv('data/hr_data.csv')
customers_df = pd.read_csv('data/customers.csv')
print(sales_df.head())      # Output: First 5 rows of sales.csv
print(hr_df.head())         # Output: First 5 rows of hr_data.csv
print(customers_df.head())  # Output: First 5 rows of customers.csv

"""2. Random Forest Feature Importance (Regression)
Use Random Forest to compute feature importance for a regression task."""
# Prepare data for regression (predict Amount in sales.csv)
sales_features = ['ProductID', 'Region']  # Example features
sales_df['Region_Encoded'] = LabelEncoder().fit_transform(sales_df['Region'])
X_sales = sales_df[['ProductID', 'Region_Encoded']]
y_sales = sales_df['Amount']
X_train, X_test, y_train, y_test = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)

# Train Random Forest
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Get feature importance
importance_reg = pd.DataFrame({
    'Feature': X_sales.columns,
    'Importance': rf_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance_reg)  # Output: Feature importance scores

"""3. Visualizing Random Forest Importance (Regression)
Plot feature importance for the regression model."""
# Bar plot for feature importance
fig = px.bar(importance_reg, x='Feature', y='Importance',
             title='Random Forest Feature Importance (Sales Amount)',
             labels={'Importance': 'Feature Importance'})
fig.update_layout(xaxis_title='Feature', yaxis_title='Importance')
fig.show()  # Output: Interactive bar plot of feature importance

"""4. Random Forest Feature Importance (Classification)
Use Random Forest for a classification task (e.g., predict Churn in customers.csv)."""
# Prepare data for classification (assume Churn is binary: 0 or 1)
customers_features = ['Age', 'Income', 'Region']
customers_df['Region_Encoded'] = LabelEncoder().fit_transform(customers_df['Region'])
X_cust = customers_df[['Age', 'Income', 'Region_Encoded']]
y_cust = customers_df['Churn']  # Assume Churn column exists
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cust, y_cust, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_c, y_train_c)

# Get feature importance
importance_clf = pd.DataFrame({
    'Feature': X_cust.columns,
    'Importance': rf_clf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance_clf)  # Output: Feature importance scores for classification

"""5. Visualizing Random Forest Importance (Classification)
Plot feature importance for the classification model."""
# Bar plot for classification feature importance
fig = px.bar(importance_clf, x='Feature', y='Importance',
             title='Random Forest Feature Importance (Customer Churn)',
             labels={'Importance': 'Feature Importance'})
fig.update_layout(xaxis_title='Feature', yaxis_title='Importance')
fig.show()  # Output: Interactive bar plot of feature importance

"""6. Permutation Importance
Permutation importance measures the impact of shuffling a feature on model performance."""
# Compute permutation importance for hr_data.csv (predict Salary)
hr_features = ['Age', 'Experience', 'Dept']
hr_df['Dept_Encoded'] = LabelEncoder().fit_transform(hr_df['Dept'])
X_hr = hr_df[['Age', 'Experience', 'Dept_Encoded']]
y_hr = hr_df['Salary']
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hr, y_hr, test_size=0.2, random_state=42)

# Train Random Forest
rf_hr = RandomForestRegressor(random_state=42)
rf_hr.fit(X_train_h, y_train_h)

# Compute permutation importance
perm_importance = permutation_importance(rf_hr, X_test_h, y_test_h, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'Feature': X_hr.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)
print(perm_df)  # Output: Permutation importance scores

"""7. Visualizing Permutation Importance
Plot permutation importance for the model."""
# Bar plot for permutation importance
fig = px.bar(perm_df, x='Feature', y='Importance',
             title='Permutation Importance (HR Salary Prediction)',
             labels={'Importance': 'Permutation Importance'})
fig.update_layout(xaxis_title='Feature', yaxis_title='Importance')
fig.show()  # Output: Interactive bar plot of permutation importance

"""8. Comparing Feature Importance Methods
Compare Random Forest and permutation importance for the same model."""
# Combine Random Forest and permutation importance for hr_data.csv
combined_importance = pd.DataFrame({
    'Feature': X_hr.columns,
    'RF_Importance': rf_hr.feature_importances_,
    'Perm_Importance': perm_importance.importances_mean
})
print(combined_importance)  # Output: Comparison of importance scores

# Plot comparison
fig = go.Figure()
fig.add_trace(go.Bar(x=combined_importance['Feature'], y=combined_importance['RF_Importance'], name='Random Forest'))
fig.add_trace(go.Bar(x=combined_importance['Feature'], y=combined_importance['Perm_Importance'], name='Permutation'))
fig.update_layout(title='Random Forest vs Permutation Importance (HR Data)',
                  xaxis_title='Feature', yaxis_title='Importance', barmode='group')
fig.show()  # Output: Grouped bar plot comparing importance methods

"""Exercises
Practice feature importance analysis with the following exercises using sales.csv, hr_data.csv, and customers.csv."""

"""Exercise 1: Load and Inspect Data
Load customers.csv and display the first 10 rows."""
customers_ex1 = pd.read_csv('data/customers.csv')
print(customers_ex1.head(10))  # Output: First 10 rows of customers.csv

"""Exercise 2: Prepare Data for Regression
Encode the Region column in sales.csv and print the first 5 rows of encoded data."""
sales_ex2 = sales_df.copy()
sales_ex2['Region_Encoded'] = LabelEncoder().fit_transform(sales_ex2['Region'])
print(sales_ex2[['Region', 'Region_Encoded']].head())  # Output: Original vs encoded Region

"""Exercise 3: Random Forest Importance (Regression)
Train a Random Forest Regressor on sales.csv to predict Amount using ProductID and Region_Encoded."""
X_ex3 = sales_ex2[['ProductID', 'Region_Encoded']]
y_ex3 = sales_ex2['Amount']
rf_ex3 = RandomForestRegressor(random_state=42)
rf_ex3.fit(X_ex3, y_ex3)
importance_ex3 = pd.DataFrame({
    'Feature': X_ex3.columns,
    'Importance': rf_ex3.feature_importances_
})
print(importance_ex3)  # Output: Feature importance scores

"""Exercise 4: Visualize Random Forest Importance
Create a bar plot for Random Forest feature importance from Exercise 3."""
fig_ex4 = px.bar(importance_ex3, x='Feature', y='Importance',
                 title='Random Forest Feature Importance (Sales Amount)')
fig_ex4.update_layout(xaxis_title='Feature', yaxis_title='Importance')
fig_ex4.show()  # Output: Bar plot of feature importance

"""Exercise 5: Random Forest Importance (Classification)
Train a Random Forest Classifier on customers.csv to predict Churn using Age, Income, and Region_Encoded."""
customers_ex5 = customers_df.copy()
customers_ex5['Region_Encoded'] = LabelEncoder().fit_transform(customers_ex5['Region'])
X_ex5 = customers_ex5[['Age', 'Income', 'Region_Encoded']]
y_ex5 = customers_ex5['Churn']
rf_ex5 = RandomForestClassifier(random_state=42)
rf_ex5.fit(X_ex5, y_ex5)
importance_ex5 = pd.DataFrame({
    'Feature': X_ex5.columns,
    'Importance': rf_ex5.feature_importances_
})
print(importance_ex5)  # Output: Feature importance scores for classification

"""Exercise 6: Visualize Classification Importance
Create a bar plot for Random Forest feature importance from Exercise 5."""
fig_ex6 = px.bar(importance_ex5, x='Feature', y='Importance',
                 title='Random Forest Feature Importance (Customer Churn)')
fig_ex6.update_layout(xaxis_title='Feature', yaxis_title='Importance')
fig_ex6.show()  # Output: Bar plot of feature importance

"""Exercise 7: Permutation Importance
Compute permutation importance for the Random Forest model on hr_data.csv (predict Salary)."""
hr_ex7 = hr_df.copy()
hr_ex7['Dept_Encoded'] = LabelEncoder().fit_transform(hr_ex7['Dept'])
X_ex7 = hr_ex7[['Age', 'Experience', 'Dept_Encoded']]
y_ex7 = hr_ex7['Salary']
X_train_ex7, X_test_ex7, y_train_ex7, y_test_ex7 = train_test_split(X_ex7, y_ex7, test_size=0.2, random_state=42)
rf_ex7 = RandomForestRegressor(random_state=42)
rf_ex7.fit(X_train_ex7, y_train_ex7)
perm_ex7 = permutation_importance(rf_ex7, X_test_ex7, y_test_ex7, n_repeats=10, random_state=42)
perm_df_ex7 = pd.DataFrame({
    'Feature': X_ex7.columns,
    'Importance': perm_ex7.importances_mean
})
print(perm_df_ex7)  # Output: Permutation importance scores

"""Exercise 8: Visualize Permutation Importance
Create a bar plot for permutation importance from Exercise 7."""
fig_ex8 = px.bar(perm_df_ex7, x='Feature', y='Importance',
                 title='Permutation Importance (HR Salary Prediction)')
fig_ex8.update_layout(xaxis_title='Feature', yaxis_title='Importance')
fig_ex8.show()  # Output: Bar plot of permutation importance

"""Exercise 9: Compare Importance Methods
Compare Random Forest and permutation importance for customers.csv (predict Churn)."""
customers_ex9 = customers_df.copy()
customers_ex9['Region_Encoded'] = LabelEncoder().fit_transform(customers_ex9['Region'])
X_ex9 = customers_ex9[['Age', 'Income', 'Region_Encoded']]
y_ex9 = customers_ex9['Churn']
X_train_ex9, X_test_ex9, y_train_ex9, y_test_ex9 = train_test_split(X_ex9, y_ex9, test_size=0.2, random_state=42)
rf_ex9 = RandomForestClassifier(random_state=42)
rf_ex9.fit(X_train_ex9, y_train_ex9)
perm_ex9 = permutation_importance(rf_ex9, X_test_ex9, y_test_ex9, n_repeats=10, random_state=42)
combined_ex9 = pd.DataFrame({
    'Feature': X_ex9.columns,
    'RF_Importance': rf_ex9.feature_importances_,
    'Perm_Importance': perm_ex9.importances_mean
})
print(combined_ex9)  # Output: Comparison of importance scores

"""Exercise 10: Visualize Comparison
Create a grouped bar plot comparing Random Forest and permutation importance from Exercise 9."""
fig_ex10 = go.Figure()
fig_ex10.add_trace(go.Bar(x=combined_ex9['Feature'], y=combined_ex9['RF_Importance'], name='Random Forest'))
fig_ex10.add_trace(go.Bar(x=combined_ex9['Feature'], y=combined_ex9['Perm_Importance'], name='Permutation'))
fig_ex10.update_layout(title='Random Forest vs Permutation Importance (Customer Churn)',
                       xaxis_title='Feature', yaxis_title='Importance', barmode='group')
fig_ex10.show()  # Output: Grouped bar plot comparing importance methods

"""Exercise 11: Feature Importance with Different Model
Train a Random Forest Regressor on financials.csv to predict Profit using Revenue and Expenses."""
financials_ex11 = financials_df.copy()
X_ex11 = financials_ex11[['Revenue', 'Expenses']]
y_ex11 = financials_ex11['Profit']
rf_ex11 = RandomForestRegressor(random_state=42)
rf_ex11.fit(X_ex11, y_ex11)
importance_ex11 = pd.DataFrame({
    'Feature': X_ex11.columns,
    'Importance': rf_ex11.feature_importances_
})
print(importance_ex11)  # Output: Feature importance scores

"""Exercise 12: Visualize Financials Importance
Create a bar plot for Random Forest feature importance from Exercise 11."""
fig_ex12 = px.bar(importance_ex11, x='Feature', y='Importance',
                  title='Random Forest Feature Importance (Financials Profit)')
fig_ex12.update_layout(xaxis_title='Feature', yaxis_title='Importance')
fig_ex12.show()  # Output: Bar plot of feature importance

"""Exercise 13: Permutation Importance with Subset
Compute permutation importance for sales.csv (predict Amount) using only ProductID."""
sales_ex13 = sales_df.copy()
X_ex13 = sales_ex13[['ProductID']]
y_ex13 = sales_ex13['Amount']
X_train_ex13, X_test_ex13, y_train_ex13, y_test_ex13 = train_test_split(X_ex13, y_ex13, test_size=0.2, random_state=42)
rf_ex13 = RandomForestRegressor(random_state=42)
rf_ex13.fit(X_train_ex13, y_train_ex13)
perm_ex13 = permutation_importance(rf_ex13, X_test_ex13, y_test_ex13, n_repeats=10, random_state=42)
perm_df_ex13 = pd.DataFrame({
    'Feature': X_ex13.columns,
    'Importance': perm_ex13.importances_mean
})
print(perm_df_ex13)  # Output: Permutation importance for ProductID

"""Exercise 14: Feature Importance with Categorical Features
Train a Random Forest Classifier on hr_data.csv to predict Dept using Age, Salary, and Experience."""
hr_ex14 = hr_df.copy()
X_ex14 = hr_ex14[['Age', 'Salary', 'Experience']]
y_ex14 = hr_ex14['Dept']
rf_ex14 = RandomForestClassifier(random_state=42)
rf_ex14.fit(X_ex14, y_ex14)
importance_ex14 = pd.DataFrame({
    'Feature': X_ex14.columns,
    'Importance': rf_ex14.feature_importances_
})
print(importance_ex14)  # Output: Feature importance scores for classification

"""Exercise 15: Visualize Categorical Importance
Create a bar plot for Random Forest feature importance from Exercise 14."""
fig_ex15 = px.bar(importance_ex14, x='Feature', y='Importance',
                  title='Random Forest Feature Importance (HR Department Prediction)')
fig_ex15.update_layout(xaxis_title='Feature', yaxis_title='Importance')
fig_ex15.show()  # Output: Bar plot of feature importance

"""Notes
- Ensure datasets (sales.csv, hr_data.csv, customers.csv) have appropriate columns (e.g., Amount, Region, Age, Income, Churn, Salary, Experience, Dept, Revenue, Expenses, Profit).
- Categorical variables (e.g., Region, Dept) must be encoded before modeling.
- Install required libraries: `pip install pandas numpy scikit-learn plotly`.
- Random Forest importance may differ from permutation importance due to model bias toward high-cardinality features.
- For advanced feature importance, consider SHAP or LIME (not covered here).
"""

if __name__ == "__main__":
    # Run all sections when script is executed
    pass