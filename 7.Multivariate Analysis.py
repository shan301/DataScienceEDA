# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 16:07:07 2025

@author: Shantanu
"""

"""Multivariate Data Analysis
Multivariate analysis involves analyzing multiple variables simultaneously to understand relationships and patterns. This script demonstrates key techniques for multivariate analysis using Python, pandas, seaborn, and scikit-learn.

1. Loading and Inspecting Data
Load a dataset and identify numerical columns for multivariate analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def load_and_inspect_data(file_path):
    df = pd.read_csv(file_path)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numerical columns for multivariate analysis: {list(numerical_cols)}")
    return df, numerical_cols

# Example usage
df, numerical_cols = load_and_inspect_data('data/sales.csv')
print(df[numerical_cols].head())

"""2. Pair Plots
Visualize pairwise relationships between numerical variables using seaborn pair plots."""
def pair_plot(df, columns):
    sns.pairplot(df[columns])
    plt.suptitle("Pair Plot of Numerical Variables", y=1.02)
    plt.show()

# Example usage
pair_plot(df, numerical_cols[:4])  # Limit to first 4 columns for clarity

"""3. Correlation Matrix
Compute and visualize a correlation matrix to understand variable relationships."""
def correlation_matrix(df, columns):
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()
    return corr

# Example usage
correlation_matrix(df, numerical_cols)

"""4. Principal Component Analysis (PCA)
Apply PCA to reduce dimensionality while preserving variance."""
def apply_pca(df, columns, n_components=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {explained_variance}")
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Result')
    plt.show()
    return pca_result

# Example usage
pca_result = apply_pca(df, numerical_cols)

"""5. t-SNE Visualization
Use t-SNE for non-linear dimensionality reduction and visualization."""
def apply_tsne(df, columns, n_components=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()
    return tsne_result

# Example usage
tsne_result = apply_tsne(df, numerical_cols)

"""6. K-Means Clustering
Apply K-Means clustering to identify groups in multivariate data."""
def kmeans_clustering(df, columns, n_clusters=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = np.nan
    df.loc[df[columns].notna().all(axis=1), 'Cluster'] = clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=clusters, palette='deep')
    plt.title('K-Means Clustering')
    plt.show()
    return df, clusters

# Example usage
df, clusters = kmeans_clustering(df, numerical_cols)

"""7. Covariance Matrix
Compute the covariance matrix to measure variable relationships."""
def covariance_matrix(df, columns):
    cov_matrix = df[columns].cov()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, cmap='viridis')
    plt.title('Covariance Matrix')
    plt.show()
    return cov_matrix

# Example usage
covariance_matrix(df, numerical_cols)

"""8. Multivariate Outlier Detection
Detect outliers using Mahalanobis distance."""
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

def mahalanobis_outliers(df, columns, threshold=3):
    data = df[columns].dropna()
    mean = data.mean()
    cov = data.cov()
    inv_cov = inv(cov)
    distances = [mahalanobis(x, mean, inv_cov) for x in data.values]
    outliers = data[distances > threshold]
    print(f"Outliers detected: {len(outliers)}")
    return outliers

# Example usage
outliers = mahalanobis_outliers(df, numerical_cols)

"""9. 3D Scatter Plot
Visualize three numerical variables in a 3D scatter plot."""
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_scatter(df, x_col, y_col, z_col):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x_col], df[y_col], df[z_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    plt.title('3D Scatter Plot')
    plt.show()

# Example usage
plot_3d_scatter(df, numerical_cols[0], numerical_cols[1], numerical_cols[2])

"""10. Feature Scaling
Standardize features to ensure fair contribution in multivariate analysis."""
def scale_features(df, columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    print(f"Scaled data sample:\n{scaled_df.head()}")
    return scaled_df

# Example usage
scaled_df = scale_features(df, numerical_cols)

"""11. Partial Correlation
Compute partial correlation to control for confounding variables."""
from pingouin import partial_corr

def partial_correlation(df, x, y, covar):
    result = partial_corr(data=df, x=x, y=y, covar=covar)
    print(f"Partial correlation between {x} and {y} controlling for {covar}:\n{result}")
    return result

# Example usage
partial_correlation(df, numerical_cols[0], numerical_cols[1], numerical_cols[2])

"""12. Canonical Correlation Analysis (CCA)
Analyze relationships between two sets of variables."""
from sklearn.cross_decomposition import CCA

def apply_cca(df, set1_cols, set2_cols, n_components=1):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[set1_cols].dropna())
    Y = scaler.fit_transform(df[set2_cols].dropna())
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    print(f"CCA correlations: {cca.score(X, Y)}")
    return X_c, Y_c

# Example usage
set1_cols = numerical_cols[:2]
set2_cols = numerical_cols[2:4]
X_c, Y_c = apply_cca(df, set1_cols, set2_cols)

"""13. Multivariate Normality Test
Test if data follows a multivariate normal distribution."""
from scipy.stats import multivariate_normal

def multivariate_normality_test(df, columns):
    data = df[columns].dropna()
    mean = data.mean()
    cov = data.cov()
    mvn = multivariate_normal(mean=mean, cov=cov)
    logpdf = mvn.logpdf(data)
    print(f"Multivariate normality log-likelihood (sample): {logpdf[:5]}")
    return logpdf

# Example usage
multivariate_normality_test(df, numerical_cols)

"""14. Interactive 3D Plot
Create an interactive 3D scatter plot using plotly."""
import plotly.express as px

def interactive_3d_plot(df, x_col, y_col, z_col):
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, title='Interactive 3D Scatter Plot')
    fig.show()

# Example usage
interactive_3d_plot(df, numerical_cols[0], numerical_cols[1], numerical_cols[2])

"""15. Variance Inflation Factor (VIF)
Calculate VIF to detect multicollinearity among variables."""
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, columns):
    data = df[columns].dropna()
    vif_data = pd.DataFrame()
    vif_data['Variable'] = columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    print(f"VIF for variables:\n{vif_data}")
    return vif_data

# Example usage
calculate_vif(df, numerical_cols)

"""Exercises for Multivariate Data Analysis
Exercise 1: Pair Plot
Write a function to create a pair plot for numerical columns."""
def ex_pair_plot(df, columns):
    sns.pairplot(df[columns])
    plt.suptitle("Exercise: Pair Plot of Numerical Variables", y=1.02)
    plt.show()

# Example usage
ex_pair_plot(df, numerical_cols[:4])

"""Exercise 2: Correlation Matrix
Write a function to compute and visualize a correlation matrix."""
def ex_correlation_matrix(df, columns):
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Exercise: Correlation Matrix")
    plt.show()
    return corr

# Example usage
ex_correlation_matrix(df, numerical_cols)

"""Exercise 3: PCA
Write a function to apply PCA and visualize the results."""
def ex_apply_pca(df, columns, n_components=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    print(f"Exercise: Explained variance ratio: {explained_variance}")
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Exercise: PCA Result')
    plt.show()
    return pca_result

# Example usage
ex_apply_pca(df, numerical_cols)

"""Exercise 4: t-SNE
Write a function to apply t-SNE and visualize the results."""
def ex_apply_tsne(df, columns, n_components=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_result = tsne.fit_transform(scaled_data)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.title('Exercise: t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()
    return tsne_result

# Example usage
ex_apply_tsne(df, numerical_cols)

"""Exercise 5: K-Means Clustering
Write a function to apply K-Means clustering and visualize clusters."""
def ex_kmeans_clustering(df, columns, n_clusters=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = np.nan
    df.loc[df[columns].notna().all(axis=1), 'Cluster'] = clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=clusters, palette='deep')
    plt.title('Exercise: K-Means Clustering')
    plt.show()
    return df, clusters

# Example usage
ex_kmeans_clustering(df, numerical_cols)

"""Exercise 6: Covariance Matrix
Write a function to compute and visualize a covariance matrix."""
def ex_covariance_matrix(df, columns):
    cov_matrix = df[columns].cov()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, cmap='viridis')
    plt.title('Exercise: Covariance Matrix')
    plt.show()
    return cov_matrix

# Example usage
ex_covariance_matrix(df, numerical_cols)

"""Exercise 7: Mahalanobis Outliers
Write a function to detect outliers using Mahalanobis distance."""
def ex_mahalanobis_outliers(df, columns, threshold=3):
    data = df[columns].dropna()
    mean = data.mean()
    cov = data.cov()
    inv_cov = inv(cov)
    distances = [mahalanobis(x, mean, inv_cov) for x in data.values]
    outliers = data[distances > threshold]
    print(f"Exercise: Outliers detected: {len(outliers)}")
    return outliers

# Example usage
ex_mahalanobis_outliers(df, numerical_cols)

"""Exercise 8: 3D Scatter Plot
Write a function to create a 3D scatter plot for three numerical variables."""
def ex_plot_3d_scatter(df, x_col, y_col, z_col):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x_col], df[y_col], df[z_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    plt.title('Exercise: 3D Scatter Plot')
    plt.show()

# Example usage
ex_plot_3d_scatter(df, numerical_cols[0], numerical_cols[1], numerical_cols[2])

"""Exercise 9: Feature Scaling
Write a function to standardize features."""
def ex_scale_features(df, columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    print(f"Exercise: Scaled data sample:\n{scaled_df.head()}")
    return scaled_df

# Example usage
ex_scale_features(df, numerical_cols)

"""Exercise 10: Partial Correlation
Write a function to compute partial correlation."""
def ex_partial_correlation(df, x, y, covar):
    result = partial_corr(data=df, x=x, y=y, covar=covar)
    print(f"Exercise: Partial correlation between {x} and {y} controlling for {covar}:\n{result}")
    return result

# Example usage
ex_partial_correlation(df, numerical_cols[0], numerical_cols[1], numerical_cols[2])

"""Exercise 11: CCA
Write a function to perform Canonical Correlation Analysis."""
def ex_apply_cca(df, set1_cols, set2_cols, n_components=1):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[set1_cols].dropna())
    Y = scaler.fit_transform(df[set2_cols].dropna())
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    print(f"Exercise: CCA correlations: {cca.score(X, Y)}")
    return X_c, Y_c

# Example usage
ex_apply_cca(df, set1_cols, set2_cols)

"""Exercise 12: Multivariate Normality Test
Write a function to test multivariate normality."""
def ex_multivariate_normality_test(df, columns):
    data = df[columns].dropna()
    mean = data.mean()
    cov = data.cov()
    mvn = multivariate_normal(mean=mean, cov=cov)
    logpdf = mvn.logpdf(data)
    print(f"Exercise: Multivariate normality log-likelihood (sample): {logpdf[:5]}")
    return logpdf

# Example usage
ex_multivariate_normality_test(df, numerical_cols)

"""Exercise 13: Interactive 3D Plot
Write a function to create an interactive 3D scatter plot."""
def ex_interactive_3d_plot(df, x_col, y_col, z_col):
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, title='Exercise: Interactive 3D Scatter Plot')
    fig.show()

# Example usage
ex_interactive_3d_plot(df, numerical_cols[0], numerical_cols[1], numerical_cols[2])

"""Exercise 14: VIF
Write a function to calculate VIF for detecting multicollinearity."""
def ex_calculate_vif(df, columns):
    data = df[columns].dropna()
    vif_data = pd.DataFrame()
    vif_data['Variable'] = columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    print(f"Exercise: VIF for variables:\n{vif_data}")
    return vif_data

# Example usage
ex_calculate_vif(df, numerical_cols)

"""Exercise 15: Elbow Method for K-Means
Write a function to determine the optimal number of clusters using the elbow method."""
def ex_elbow_method(df, columns, max_clusters=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns].dropna())
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title('Exercise: Elbow Method for K-Means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

# Example usage
ex_elbow_method(df, numerical_cols)