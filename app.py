import streamlit as st
from components import eda
import pandas as pd

# Set up the Streamlit app
st.set_page_config(page_title="SIMPLE DATA MINING", layout="wide")

# App Title
st.title("Simple Data Mining")

# File Upload Section
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Display basic information if a file is uploaded
if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    with st.expander(f"Basic EDA of {uploaded_file.name}"):
        eda.display_basic_info(data)
    with st.expander("Summary statistics"):
        eda.display_summary_statistics(data)
    with st.expander("Correlation Matrix"):
        eda.display_correlation_matrix(data)
    with st.expander("Outlier Detection"):
        eda.display_outlier_detection(data)
    with st.expander("Pairplot"):
        eda.display_pairplot(data)
