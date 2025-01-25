import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def filter_based_correlation(data, threshold):
    """
    Filters features based on correlation with the target variable.

    Parameters:
    - data: pd.DataFrame
    - threshold: float, correlation threshold for feature selection
    """
    correlation_matrix = data.corr()
    st.write("Correlation Matrix:")
    st.write(correlation_matrix)
    high_corr = correlation_matrix[(correlation_matrix.abs() > threshold)]
    st.write(f"Features with correlation above {threshold}:")
    st.write(high_corr)

def wrapper_based_rfe(data, target_column, num_features):
    """
    Performs recursive feature elimination (RFE) to select the top features.

    Parameters:
    - data: pd.DataFrame
    - target_column: str, the target variable
    - num_features: int, number of features to select
    """
    features = data.drop(columns=[target_column])
    target = data[target_column]
    model = RandomForestClassifier()
    selector = RFE(estimator=model, n_features_to_select=num_features, step=1)
    selector = selector.fit(features, target)

    selected_features = [feature for feature, selected in zip(features.columns, selector.support_) if selected]
    st.write("Selected Features using RFE:")
    st.write(selected_features)

def feature_extraction_pca(data, n_components):
    """
    Performs Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters:
    - data: pd.DataFrame
    - n_components: int, number of principal components to retain
    """
    numeric_features = data.select_dtypes(include=['number'])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(numeric_features)
    explained_variance = pca.explained_variance_ratio_
    st.write("Explained Variance Ratio of Components:")
    st.write(explained_variance)
    st.write("Principal Components:")
    st.write(pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)]))
