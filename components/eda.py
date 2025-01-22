import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def display_basic_info(data):
    
    # Dataset Shape
    st.write(f"**Shape of the dataset:** {data.shape[0]} rows, {data.shape[1]} columns")

    # Column names and Data Types
    st.write("**Column names and Data Types:**")
    dtypes_df = pd.DataFrame(data.dtypes).T  # Convert to DataFrame and Transpose
    dtypes_df.index = ['dtype']  # Rename the index for clarity

    st.write( dtypes_df)

    # Missing Values
    missing_values = pd.DataFrame(data.isnull().sum()).T
    missing_values.index = ['Missing Values']
    st.write("**Missing Values Per Column:**")
    st.write(missing_values[missing_values > 0])

    # Duplicate Rows
    duplicates = data.duplicated().sum()
    st.write(f"**Number of Duplicate Rows:** {duplicates}")

    # Unique Values
    uniqe_data = pd.DataFrame(data.nunique()).T
    uniqe_data.index = ["Unique Data"]
    st.write("**Unique Values Per Column:**")
    st.write(uniqe_data)

def display_summary_statistics(data):

    # Summary Statistics for Numerical Columns
    st.write("### Summary Statistics (Numerical Columns)")
    numerical_summary = data.describe().transpose()
    st.write(numerical_summary)

    # Summary Statistics for Categorical Columns
    st.write("### Summary Statistics (Categorical Columns)")
    categorical_columns = data.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        categorical_summary = data[categorical_columns].describe().transpose()
        st.write(categorical_summary)
    else:
        st.write("No categorical columns found.")

def display_outlier_detection(data):

    # Identify numerical columns
    numerical_columns = data.select_dtypes(include=['number']).columns

    selected_columns = st.multiselect("Select columns for outlier detection", data.select_dtypes(include=['number']).columns)
    for col in selected_columns:
        st.write(f"Boxplot for `{col}`:")
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        st.pyplot(fig)