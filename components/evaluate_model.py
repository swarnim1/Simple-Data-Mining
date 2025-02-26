import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)

def display_performance_metrics(model, X_test, y_test, y_pred, is_continuous):
    """
    Displays performance metrics for regression or classification models
    """
   
    if is_continuous:
        # Regression Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Regression Metrics")
        st.text(f"Mean Squared Error: {mse:.4f}")
        st.text(f"Mean Absolute Error: {mae:.4f}")
        st.text(f"R-squared: {r2:.4f}")

        # Regression Visualizations
        st.subheader("Visualizations")
        
        # Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        st.pyplot(plt)

        # Residual Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, bins=30)
        plt.title("Residual Plot")
        plt.xlabel("Residuals")
        st.pyplot(plt)

    else:
        # Classification Metrics
        accuracy = (y_test == y_pred).mean()
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.subheader("Classification Metrics")
        st.text(f"Accuracy: {accuracy:.4f}")
        st.text(f"Precision: {precision:.4f}")
        st.text(f"Recall: {recall:.4f}")
        st.text(f"F1-Score: {f1:.4f}")

        # Classification Visualizations
        st.subheader("Visualizations")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot(plt)

        # ROC Curve (for binary classification)
        if len(set(y_test)) == 2:  # Ensure it's binary classification
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], 'k--', color="red")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend(loc="lower right")
            st.pyplot(plt)
