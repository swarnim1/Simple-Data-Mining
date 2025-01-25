import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np

def display_basic_info(data):
    
    # Dataset Shape
    st.header("Basic Information")
    st.write(f"**Shape of the dataset:** {data.shape[0]} rows, {data.shape[1]} columns")

    # Column names and Data Types
    st.write("**Column names and Data Types:**")
    dtypes_df = pd.DataFrame(data.dtypes).transpose()  # Convert to DataFrame and Transpose
    dtypes_df.index = ['dtype']  # Rename the index for clarity

    st.write( dtypes_df)

    # Missing Values
    missing_values = pd.DataFrame(data.isnull().sum()).transpose()
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



# Display pairplot with customization options in a sidebar
def display_pairplot(data):
    st.header("Pairplot")
    
    
    # Sidebar for customization options
    with st.sidebar:
        st.subheader("Pairplot Customization Options")
        
        # Select numerical columns
        numerical_columns = list(data.select_dtypes(include=['number']).columns)
        
        # Multiselect widget for column selection
        selected_columns = st.multiselect(
            "Select columns for pairplot",
            numerical_columns,
            default=numerical_columns[:2] if len(numerical_columns) > 1 else numerical_columns
        )
        
        # Additional options
        hue = st.selectbox("Hue (categorical column)", [None] + list(data.select_dtypes(include=['category', 'object']).columns), index=0)
        kind = st.selectbox("Kind of plot", ['scatter', 'kde', 'hist', 'reg'], index=0)
        diag_kind = st.selectbox("Kind of plot for diagonal", ['auto', 'hist', 'kde', None], index=0)
        corner = st.checkbox("Show only lower triangle (corner plot)", value=False)
        height = st.slider("Plot height (in inches)", min_value=1.0, max_value=5.0, value=2.5)
        aspect = st.slider("Aspect ratio", min_value=0.5, max_value=2.0, value=1.0)
        
        # Advanced keyword options
        plot_kws = st.text_area("Additional bivariate plot options (as dictionary)", "{}")
        diag_kws = st.text_area("Additional diagonal plot options (as dictionary)", "{}")
    
    # Main area for the pairplot
    if len(numerical_columns) > 1:
        if selected_columns:
            try:
                # Convert keyword options from text to dictionaries
                plot_kws = eval(plot_kws)
                diag_kws = eval(diag_kws)
                
                
                # Create pairplot
                fig = sns.pairplot(
                    data,
                    vars=selected_columns,
                    hue=hue,
                    kind=kind,
                    diag_kind=diag_kind,
                    corner=corner,
                    height=height,
                    aspect=aspect,
                    plot_kws=plot_kws if isinstance(plot_kws, dict) else None,
                    diag_kws=diag_kws if isinstance(diag_kws, dict) else None
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating pairplot: {e}")
        else:
            st.warning("Please select at least one numerical column for the pairplot.")
    else:
        st.warning("Not enough numerical columns in the dataset to create a pairplot.")
    
       






def display_scatterplot_old(data):
    st.header("Scatter Plot")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)  # Ensure it's a Python list
    
    # Check if there are at least two numerical columns
    if len(numerical_columns) > 1:
        # Dropdowns for selecting x and y axes
        x_col = st.selectbox("Select X-axis column", numerical_columns)
        y_col = st.selectbox("Select Y-axis column", numerical_columns)
        
        # Create scatter plot
        fig, ax = plt.subplots()
        ax.scatter(data[x_col], data[y_col], alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        st.pyplot(fig)
    else:
        st.warning("Not enough numerical columns in the dataset to create a scatter plot.")

def display_scatterplot(data):
    st.header("Scatter Plot")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    categorical_columns = list(data.select_dtypes(include=['category', 'object']).columns)
    
    # Check if there are at least two numerical columns
    if len(numerical_columns) > 1:
        # Dropdowns for selecting x and y axes
        x_col = st.selectbox("Select X-axis column", numerical_columns)
        y_col = st.selectbox("Select Y-axis column", numerical_columns)
        
        # Additional customization options
        with st.sidebar:
            st.subheader("Scatter Plot Customization")
            hue = st.selectbox("Hue (categorical column)", [None] + categorical_columns)
            size_col = st.selectbox("Size (numerical column)", [None] + numerical_columns)
            style_col = st.selectbox("Style (categorical column)", [None] + categorical_columns)
            palette = st.selectbox("Color Palette", [None, "deep", "bright", "dark", "colorblind", "muted", "pastel"])
            alpha = st.slider("Point Transparency (Alpha)", min_value=0.1, max_value=1.0, value=0.7)
            size = st.slider("Point Size (when no size column)", min_value=5, max_value=100, value=20)
        
        # Create scatter plot
        try:
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=data,
                x=x_col,
                y=y_col,
                hue=hue,
                size=size_col,
                style=style_col,
                palette=palette,
                alpha=alpha,
                sizes=(size, size * 2),  # Scale size
                ax=ax
            )
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            st.pyplot(fig)
            # Add download button for the plot
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Scatter Plot as PNG",
                data=buffer,
                file_name="scatter_plot.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error creating scatter plot: {e}")
    else:
        st.warning("Not enough numerical columns in the dataset to create a scatter plot.")



def display_histogram(data):
    st.header("Histogram")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    
    # Check if there are numerical columns
    if numerical_columns:
        # Dropdown for selecting the column
        selected_column = st.selectbox("Select column for histogram", numerical_columns)
        
        # Additional customization options
        with st.sidebar:
            st.subheader("Histogram Customization")
            bins = st.slider("Number of Bins", min_value=5, max_value=100, value=20)
            color = st.color_picker("Pick a Bar Color", "#3498db")
            alpha = st.slider("Bar Transparency (Alpha)", min_value=0.1, max_value=1.0, value=0.7)
            kde = st.checkbox("Show Kernel Density Estimate (KDE)", value=False)
        
        # Create histogram
        try:
            fig, ax = plt.subplots()
            sns.histplot(
                data[selected_column],
                bins=bins,
                kde=kde,
                color=color,
                alpha=alpha,
                ax=ax
            )
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram: {selected_column}")
            st.pyplot(fig)
            
            # Add download button for the plot
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Histogram as PNG",
                data=buffer,
                file_name=f"histogram_{selected_column}.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error creating histogram: {e}")
    else:
        st.warning("No numerical columns in the dataset to create a histogram.")



def display_boxplot(data):
    st.header("Box Plot")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    categorical_columns = list(data.select_dtypes(include=['category', 'object']).columns)
    
    # Check if there are numerical columns
    if numerical_columns:
        # Dropdown for selecting the column
        selected_column = st.selectbox("Select numerical column for box plot", numerical_columns)
        
        # Additional customization options
        with st.sidebar:
            st.subheader("Box Plot Customization")
            group_by = st.selectbox("Group by (categorical column)", [None] + categorical_columns)
            color = st.color_picker("Pick a Box Color", "#3498db")
            show_points = st.checkbox("Show Individual Points (Swarmplot Overlay)", value=False)
            point_alpha = st.slider("Point Transparency (Alpha)", min_value=0.1, max_value=1.0, value=0.7) if show_points else None
        
        # Create box plot
        try:
            fig, ax = plt.subplots()
            sns.boxplot(
                data=data,
                x=group_by if group_by else None,
                y=selected_column,
                color=color,
                ax=ax
            )
            
            if show_points:
                sns.swarmplot(
                    data=data,
                    x=group_by if group_by else None,
                    y=selected_column,
                    color="black",
                    alpha=point_alpha,
                    ax=ax
                )
            
            ax.set_title(f"Box Plot: {selected_column}" + (f" by {group_by}" if group_by else ""))
            st.pyplot(fig)
            
            # Add download button for the plot
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Box Plot as PNG",
                data=buffer,
                file_name=f"boxplot_{selected_column}.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error creating box plot: {e}")
    else:
        st.warning("No numerical columns in the dataset to create a box plot.")



def display_heatmap(data):
    st.header("Heatmap")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    
    # Check if there are numerical columns
    if len(numerical_columns) > 1:
        # Sidebar for customization options
        with st.sidebar:
            st.subheader("Heatmap Customization")
            
            # Select correlation method
            correlation_method = st.selectbox(
                "Correlation Method",
                ["pearson (linear)", "kendall (rank)", "spearman (rank)"],
                index=0
            )
            
            # Additional options
            annot = st.checkbox("Show Annotations", value=True)
            fmt = st.selectbox("Annotation Format", ["0.2f", "0.1f", "d"], index=0)
            cmap = st.selectbox(
                "Color Palette",
                ["coolwarm", "viridis", "plasma", "inferno", "magma", "cividis", "rocket", "vlag"],
                index=0
            )
            linewidths = st.slider("Line Widths Between Cells", min_value=0.0, max_value=2.0, value=0.5)
            cbar = st.checkbox("Show Color Bar", value=True)
        
        # Map selected method to valid options
        method_mapping = {
            "pearson (linear)": "pearson",
            "kendall (rank)": "kendall",
            "spearman (rank)": "spearman"
        }
        selected_method = method_mapping[correlation_method]
        
        # Calculate correlation matrix
        corr_matrix = data[numerical_columns].corr(method=selected_method)
        
        # Create heatmap
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_matrix,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                linewidths=linewidths,
                cbar=cbar,
                ax=ax
            )
            ax.set_title(f"Heatmap ({correlation_method.split(' ')[0]} Correlation)")
            st.pyplot(fig)
            
            # Add download button for the plot
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Heatmap as PNG",
                data=buffer,
                file_name="heatmap.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
    else:
        st.warning("Not enough numerical columns in the dataset to create a heatmap.")


def display_barplot(data):
    st.header("Bar Plot")
    
    # Select categorical and numerical columns
    categorical_columns = list(data.select_dtypes(include=['category', 'object']).columns)
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    
    # Check if there are categorical and numerical columns
    if categorical_columns and numerical_columns:
        # Dropdowns for selecting categorical and numerical columns
        x_col = st.selectbox("Select X-axis (categorical)", categorical_columns)
        y_col = st.selectbox("Select Y-axis (numerical)", numerical_columns)
        
        # Sidebar for customization options
        with st.sidebar:
            st.subheader("Bar Plot Customization")
            estimator = st.selectbox(
                "Estimation Method",
                ["mean", "sum", "count"],
                index=0
            )
            hue = st.selectbox("Hue (categorical)", [None] + categorical_columns)
            palette = st.selectbox(
                "Color Palette",
                ["deep", "bright", "dark", "colorblind", "muted", "pastel"],
                index=0
            )
            orient = st.radio("Bar Orientation", ["Vertical", "Horizontal"], index=0)
            dodge = st.checkbox("Separate Bars by Hue (Dodge)", value=True)
            ci = st.slider("Confidence Interval (%)", min_value=0, max_value=100, value=95)
        
        # Define the orientation
        orientation = 'v' if orient == 'Vertical' else 'h'
        
        # Map estimator names to actual functions
        estimator_func = {
            "mean": np.mean,
            "sum": np.sum,
            "count": len
        }[estimator]
        
        # Create bar plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=data,
                x=x_col if orientation == 'v' else y_col,
                y=y_col if orientation == 'v' else x_col,
                hue=hue,
                palette=palette,
                estimator=estimator_func,
                ci=ci,
                dodge=dodge,
                orient=orientation,
                ax=ax
            )
            ax.set_title(f"Bar Plot: {y_col} vs {x_col}" + (f" by {hue}" if hue else ""))
            st.pyplot(fig)
            
            # Add download button for the plot
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Bar Plot as PNG",
                data=buffer,
                file_name=f"barplot_{x_col}_vs_{y_col}.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error creating bar plot: {e}")
    else:
        st.warning("The dataset must have both categorical and numerical columns to create a bar plot.")

def display_lineplot(data):
    st.header("Line Plot")
    
    # Select numerical and categorical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    categorical_columns = list(data.select_dtypes(include=['category', 'object']).columns)
    
    # Check if there are numerical columns
    if numerical_columns:
        # Dropdowns for selecting X and Y axes
        x_col = st.selectbox("Select X-axis (categorical or numerical)", numerical_columns + categorical_columns)
        y_col = st.selectbox("Select Y-axis (numerical)", numerical_columns)
        
        # Sidebar for customization options
        with st.sidebar:
            st.subheader("Line Plot Customization")
            hue = st.selectbox("Hue (categorical)", [None] + categorical_columns)
            style = st.selectbox("Line Style (categorical)", [None] + categorical_columns)
            markers = st.checkbox("Show Markers on Line", value=True)
            palette = st.selectbox(
                "Color Palette",
                ["deep", "bright", "dark", "colorblind", "muted", "pastel"],
                index=0
            )
            linewidth = st.slider("Line Width", min_value=1, max_value=10, value=2)
        
        # Create line plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(
                data=data,
                x=x_col,
                y=y_col,
                hue=hue,
                style=style,
                markers=markers,
                palette=palette,
                linewidth=linewidth,
                ax=ax
            )
            ax.set_title(f"Line Plot: {y_col} vs {x_col}" + (f" by {hue}" if hue else ""))
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
            
            # Add download button for the plot
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Line Plot as PNG",
                data=buffer,
                file_name=f"lineplot_{x_col}_vs_{y_col}.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error creating line plot: {e}")
    else:
        st.warning("The dataset must have at least one numerical column to create a line plot.")


def display_violinplot(data):
    st.header("Violin Plot")
    
    # Select numerical and categorical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    categorical_columns = list(data.select_dtypes(include=['category', 'object']).columns)
    
    # Check if there are numerical columns
    if numerical_columns and categorical_columns:
        # Dropdowns for selecting X and Y axes
        y_col = st.selectbox("Select Y-axis (numerical)", numerical_columns)
        x_col = st.selectbox("Select X-axis (categorical)", categorical_columns)
        
        # Sidebar for customization options
        with st.sidebar:
            st.subheader("Violin Plot Customization")
            hue = st.selectbox("Hue (categorical)", [None] + categorical_columns)
            split = st.checkbox("Split Violin by Hue", value=False) if hue else False
            inner = st.selectbox("Inner Data Representation", ["box", "quartile", "point", "stick", None], index=0)
            palette = st.selectbox(
                "Color Palette",
                ["deep", "bright", "dark", "colorblind", "muted", "pastel"],
                index=0
            )
            scale = st.selectbox("Scale Method", ["area", "count", "width"], index=0)
        
        # Create violin plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(
                data=data,
                x=x_col,
                y=y_col,
                hue=hue,
                split=split,
                inner=inner,
                palette=palette,
                scale=scale,
                ax=ax
            )
            ax.set_title(f"Violin Plot: {y_col} by {x_col}" + (f" with Hue: {hue}" if hue else ""))
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
            
            # Add download button for the plot
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Violin Plot as PNG",
                data=buffer,
                file_name=f"violinplot_{x_col}_vs_{y_col}.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error creating violin plot: {e}")
    else:
        st.warning("The dataset must have both numerical and categorical columns to create a violin plot.")





def display_timeseries_plot(data):
    st.header("Time-Series Plot")
    
    # Allow user to select a column to convert to datetime
    st.subheader("Datetime Configuration")
    datetime_columns = list(data.select_dtypes(include=['datetime', 'object']).columns)
    datetime_col = st.selectbox("Select a column for the time axis (or convert to datetime)", datetime_columns)
    
    if datetime_col:
        # Conversion option for the selected column
        if data[datetime_col].dtype != 'datetime64[ns]':
            convert_to_datetime = st.checkbox(f"Convert '{datetime_col}' to datetime", value=True)
            if convert_to_datetime:
                data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
                data = data.dropna(subset=[datetime_col])  # Drop rows with invalid datetime
                st.success(f"'{datetime_col}' converted to datetime format.")
    
    # Select numerical columns
    numerical_columns = list(data.select_dtypes(include=['number']).columns)
    if not numerical_columns:
        st.warning("The dataset must have at least one numerical column to create a time-series plot.")
        return
    
    # Select numerical column for the Y-axis
    value_col = st.selectbox("Select Value Column (numerical)", numerical_columns)
    
    # Sidebar for customization options
    with st.sidebar:
        st.subheader("Time-Series Plot Customization")
        hue = st.selectbox("Hue (categorical)", [None] + list(data.select_dtypes(include=['category', 'object']).columns))
        style = st.selectbox("Line Style (categorical)", [None] + list(data.select_dtypes(include=['category', 'object']).columns))
        markers = st.checkbox("Show Markers on Line", value=False)
        palette = st.selectbox(
            "Color Palette",
            ["deep", "bright", "dark", "colorblind", "muted", "pastel"],
            index=0
        )
        linewidth = st.slider("Line Width", min_value=1, max_value=10, value=2)
        show_rolling = st.checkbox("Show Rolling Average", value=False)
        window_size = st.slider("Rolling Window Size", min_value=2, max_value=30, value=7) if show_rolling else None
    
    # Add rolling average column if required
    if show_rolling:
        data[f"{value_col}_rolling"] = data[value_col].rolling(window=window_size).mean()
    
    # Create time-series plot
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=data,
            x=datetime_col,
            y=value_col,
            hue=hue,
            style=style,
            markers=markers,
            palette=palette,
            linewidth=linewidth,
            ax=ax
        )
        
        # Plot rolling average if enabled
        if show_rolling:
            sns.lineplot(
                data=data,
                x=datetime_col,
                y=f"{value_col}_rolling",
                ax=ax,
                label=f"{window_size}-point Rolling Average",
                color="orange",
                linestyle="--"
            )
        
        ax.set_title(f"Time-Series Plot: {value_col} over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(value_col)
        st.pyplot(fig)
        
        # Add download button for the plot
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        st.download_button(
            label="Download Time-Series Plot as PNG",
            data=buffer,
            file_name=f"timeseries_{value_col}.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error creating time-series plot: {e}")
