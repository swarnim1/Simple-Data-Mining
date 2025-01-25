import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    RFE,
    mutual_info_classif,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Filter-based feature selection
def filter_based_methods(data, target_column, threshold):
    st.header("Filter-Based Feature Selection")
    
    # Select numeric columns for feature selection
    numeric_data = data.select_dtypes(include=[float, int])
    selected_features = [col for col in numeric_data.columns if col != target_column]
    target = data[target_column]
    
    # Select filter technique
    technique = st.selectbox("Select Filter-Based Technique", ["Correlation", "Chi-Square", "ANOVA", "Mutual Information"])
    
    if technique == "Correlation":
        # Compute the correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Filter the correlation matrix based on the threshold
        high_correlation = correlation_matrix[correlation_matrix > threshold]
        
        st.write(f"Filtered Correlation Matrix (Threshold: {threshold}):")
        st.write(high_correlation)

    elif technique == "Chi-Square":
        # Select K best features using the Chi-Square method
        selector = SelectKBest(score_func=chi2, k="all")
        selector.fit(data[selected_features], target)
        
        st.write("Chi-Square Scores:")
        st.write(pd.DataFrame({"Feature": selected_features, "Score": selector.scores_}))

    elif technique == "ANOVA":
        # Select K best features using ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k="all")
        selector.fit(data[selected_features], target)
        
        st.write("ANOVA F-Scores:")
        st.write(pd.DataFrame({"Feature": selected_features, "Score": selector.scores_}))

    elif technique == "Mutual Information":
        # Select features using Mutual Information
        scores = mutual_info_classif(data[selected_features], target)
        
        st.write("Mutual Information Scores:")
        st.write(pd.DataFrame({"Feature": selected_features, "Score": scores}))


# Wrapper-based feature selection
def wrapper_based_methods(data, target_column, num_features):
    st.header("Wrapper-Based Feature Selection")
    
    # Select numeric features and target column
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    selected_features = [col for col in numeric_columns if col != target_column]
    target = data[target_column]
    
    model = RandomForestClassifier()
    selector = RFE(estimator=model, n_features_to_select=num_features, step=1)
    selector.fit(data[selected_features], target)
    
    st.write("Selected Features:")
    st.write([feature for feature, selected in zip(selected_features, selector.support_) if selected])


# Feature extraction methods
def feature_extraction_methods(data, n_components):
    st.header("Feature Extraction")
    
    # Select numeric columns
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    selected_features = numeric_columns
    
    method = st.selectbox("Select Feature Extraction Technique", ["PCA", "LDA"])
    
    if method == "PCA":
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data[selected_features])
        st.write("Explained Variance Ratio:")
        st.write(pca.explained_variance_ratio_)

    elif method == "LDA":
        target_column = st.selectbox("Select Target Column for LDA", data.select_dtypes(include=["number"]).columns)
        lda = LDA(n_components=1)
        lda_transformed = lda.fit_transform(data[selected_features], data[target_column])
        st.write("LDA Transformed Data:")
        st.write(pd.DataFrame(lda_transformed, columns=["LDA1"]))
