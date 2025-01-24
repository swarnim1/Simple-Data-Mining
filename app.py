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

st.sidebar.divider()

# Display basic information if a file is uploaded
if uploaded_file is None:
    st.text("Please Upload CSV file")
elif uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    with st.expander(f"Basic EDA of {uploaded_file.name}"):
        eda.display_basic_info(data)
    with st.expander("Visualizations"):
        visualization_type = st.selectbox(
            "Select Visualization Type",
            [None,"Pairplot", "Scatter Plot", "Histogram", "Box Plot","Heatmap" , "Bar Plot","Line Plot","Violin Plot","Time-Series Plot"]
        )
        if visualization_type == None:
            st.text("Choose a plot")
        if visualization_type == "Pairplot":
            eda.display_pairplot(data)
        elif visualization_type == "Scatter Plot":
            eda.display_scatterplot(data)
        elif visualization_type == "Histogram":
            eda.display_histogram(data)
        elif visualization_type == "Box Plot":
            eda.display_boxplot(data)
        elif visualization_type == "Heatmap":
            eda.display_heatmap(data)
        elif visualization_type == "Bar Plot":
            eda.display_barplot(data)
        elif visualization_type == "Line Plot":
            eda.display_lineplot(data)
        elif visualization_type == "Violin Plot":
            eda.display_violinplot(data)
        elif visualization_type == "Time-Series Plot":
            eda.display_timeseries_plot(data)

