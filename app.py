import streamlit as st
from components import eda, preprocessing
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
    st.text("Please Upload a CSV file")
else:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    with st.expander(f"Basic EDA of {uploaded_file.name}"):
        eda.display_basic_info(data)

    with st.expander("Visualizations"):
        visualization_type = st.selectbox(
            "Select Visualization Type",
            [None, "Pairplot", "Scatter Plot", "Histogram", "Heatmap", "Bar Plot", "Line Plot", "Violin Plot", "Time-Series Plot"]
        )
        if visualization_type:
            if visualization_type == "Pairplot":
                eda.display_pairplot(data)
            elif visualization_type == "Scatter Plot":
                eda.display_scatterplot(data)
            elif visualization_type == "Histogram":
                eda.display_histogram(data)
            elif visualization_type == "Heatmap":
                st.subheader("Heatmap Settings")
                correlation_method = st.selectbox("Select Correlation Method", ["pearson", "spearman", "kendall"], index=0)
                eda.display_heatmap(data, correlation_method)
            elif visualization_type == "Bar Plot":
                eda.display_barplot(data)
            elif visualization_type == "Line Plot":
                eda.display_lineplot(data)
            elif visualization_type == "Violin Plot":
                eda.display_violinplot(data)
            elif visualization_type == "Time-Series Plot":
                eda.display_timeseries_plot(data)

    with st.expander("Outlier Detection"):
        st.subheader("Outlier Detection Techniques")
        detection_method = st.selectbox(
            "Select Outlier Detection Technique",
            [
                "Outlier Detection: Box Plot",
                "Outlier Detection: Z-Score",
                "Outlier Detection: IQR Method",
                "Outlier Detection: Isolation Forest"
            ],
            index=0
        )
        eda.outlier_detection(data, detection_method)

    with st.expander("Feature Engineering"):
        st.subheader("Feature Engineering Techniques")

        # Categorized techniques
        feature_eng_method = st.selectbox(
            "Select Feature Engineering Technique",[
            "Filter-Based Techniques",
            "Wrapper-Based Techniques",
            "Feature Extraction: PCA",
            "Feature Extraction: LDA"
            ],
            index=0
            )

        if feature_eng_method == "Filter-Based Techniques":
            st.subheader("Filter-Based Techniques")
            target_column = st.selectbox("Select Target Column", data.select_dtypes(include=["number"]).columns)
            threshold = st.slider(
                "Set Threshold (for Correlation)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01
            )
            # Only keep this one dropdown for Filter-Based Technique
            # filter_technique = st.selectbox(
            #   "Select Filter-Based Technique",
            #    ["Correlation", "Chi-Square", "ANOVA", "Mutual Information"],
            #    index=0
            #    )
            preprocessing.filter_based_methods(data, target_column, threshold)

        elif feature_eng_method == "Wrapper-Based Techniques":
            st.subheader("Wrapper-Based Techniques")
            target_column = st.selectbox(
                "Select Target Column",
                data.select_dtypes(include=["number"]).columns
            )
            num_features = st.slider(
                "Number of Features to Select",
                min_value=1,
                max_value=len(data.columns) - 1,
                value=5
            )
            preprocessing.wrapper_based_methods(data, target_column, num_features)

        elif feature_eng_method == "Feature Extraction: PCA":
            st.subheader("Feature Extraction: PCA")
            n_components = st.slider(
                "Number of Principal Components",
                min_value=1,
                max_value=min(len(data.columns), len(data)),
                value=2
            )
            preprocessing.feature_extraction_methods(data, n_components)

        elif feature_eng_method == "Feature Extraction: LDA":
            st.subheader("Feature Extraction: LDA")
            target_column = st.selectbox(
                "Select Target Column",
                data.select_dtypes(include=["object", "category"]).columns
            )
            n_components = st.slider(
                "Number of Components",
                min_value=1,
                max_value=min(len(data.columns), len(data)),
                value=2
            )
            preprocessing.feature_extraction_methods(data, n_components)

        else:
            st.text("Select a feature engineering technique")
