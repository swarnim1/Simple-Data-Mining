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

def filter_based_methods(data, target_column):
    st.header("Filter-Based Feature Selection")
    
    # Choose technique
    technique = st.selectbox("Select Filter-Based Technique", ["Correlation", "Chi-Square", "ANOVA", "Mutual Information"])
    
    # Select target column and numeric features
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    selected_features = [col for col in numeric_columns if col != target_column]
    target = data[target_column]
    
    if technique == "Correlation":
        correlation = data[selected_features].corrwith(target)
        st.write("Correlation with Target Variable:")
        st.write(correlation)
    elif technique == "Chi-Square":
        selector = SelectKBest(score_func=chi2, k="all")
        selector.fit(data[selected_features], target)
        st.write("Chi-Square Scores:")
        st.write(pd.DataFrame({"Feature": selected_features, "Score": selector.scores_}))
    elif technique == "ANOVA":
        selector = SelectKBest(score_func=f_classif, k="all")
        selector.fit(data[selected_features], target)
        st.write("ANOVA F-Scores:")
        st.write(pd.DataFrame({"Feature": selected_features, "Score": selector.scores_}))
    elif technique == "Mutual Information":
        scores = mutual_info_classif(data[selected_features], target)
        st.write("Mutual Information Scores:")
        st.write(pd.DataFrame({"Feature": selected_features, "Score": scores}))


def wrapper_based_methods(data, target_column):
    st.header("Wrapper-Based Feature Selection")
    
    # Choose wrapper method
    method = st.selectbox("Select Wrapper-Based Technique", ["RFE", "Forward Selection"])
    
    # Select target column and numeric features
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    selected_features = [col for col in numeric_columns if col != target_column]
    target = data[target_column]
    
    if method == "RFE":
        model = RandomForestClassifier()
        selector = RFE(estimator=model, n_features_to_select=5, step=1)
        selector.fit(data[selected_features], target)
        st.write("Selected Features:")
        st.write([feature for feature, selected in zip(selected_features, selector.support_) if selected])
    elif method == "Forward Selection":
        st.write("Forward Selection is under development")


def feature_extraction_methods(data, target_column):
    st.header("Feature Extraction")
    
    # Choose extraction method
    method = st.selectbox("Select Feature Extraction Technique", ["PCA", "LDA"])
    
    # Select target column and numeric features
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    selected_features = [col for col in numeric_columns if col != target_column]
    target = data[target_column]
    
    if method == "PCA":
        n_components = st.slider("Number of Components", 1, len(selected_features))
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data[selected_features])
        st.write("Explained Variance Ratio:")
        st.write(pca.explained_variance_ratio_)
    elif method == "LDA":
        lda = LDA(n_components=1)
        lda_transformed = lda.fit_transform(data[selected_features], target)
        st.write("LDA Transformed Data:")
        st.write(pd.DataFrame(lda_transformed, columns=["LDA1"]))
