import streamlit as st
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Function to dynamically configure hyperparameters
def configure_hyperparameters(model_name, tuning_method):
    st.sidebar.subheader("Hyperparameter Configuration")
    param_grid = {}

    if model_name == "Random Forest Regressor" or model_name == "Random Forest Classifier":
        st.sidebar.text("Random Forest Hyperparameters")
        if tuning_method in ["Grid Search", "Random Search"]:
            # Allow user to add hyperparameters dynamically
            if st.sidebar.checkbox("n_estimators (Number of Trees)"):
                param_grid['n_estimators'] = st.sidebar.text_input(
                    "Values for n_estimators (comma-separated)", "10,50,100"
                ).split(",")
                param_grid['n_estimators'] = [int(val) for val in param_grid['n_estimators']]

            if st.sidebar.checkbox("max_depth (Maximum Depth)"):
                param_grid['max_depth'] = st.sidebar.text_input(
                    "Values for max_depth (comma-separated)", "5,10,20"
                ).split(",")
                param_grid['max_depth'] = [int(val) for val in param_grid['max_depth']]

            if st.sidebar.checkbox("min_samples_split"):
                param_grid['min_samples_split'] = st.sidebar.text_input(
                    "Values for min_samples_split (comma-separated)", "2,5,10"
                ).split(",")
                param_grid['min_samples_split'] = [int(val) for val in param_grid['min_samples_split']]

    elif model_name == "Logistic Regression":
        st.sidebar.text("Logistic Regression Hyperparameters")
        if tuning_method in ["Grid Search", "Random Search"]:
            if st.sidebar.checkbox("C (Inverse Regularization Strength)"):
                param_grid['C'] = st.sidebar.text_input(
                    "Values for C (comma-separated)", "0.1,1.0,10"
                ).split(",")
                param_grid['C'] = [float(val) for val in param_grid['C']]

            if st.sidebar.checkbox("solver"):
                param_grid['solver'] = st.sidebar.multiselect(
                    "Values for solver", ["lbfgs", "liblinear", "sag", "saga"], default=["lbfgs"]
                )

    elif model_name == "Linear Regression":
        st.sidebar.text("Linear Regression does not require hyperparameters.")
        param_grid = {}  # No hyperparameters for Linear Regression.

    return param_grid


# Function to run Grid Search or Random Search with configurable CV and n_iter
def run_hyperparameter_search(model, param_grid, X_train, y_train, tuning_method ,cv, n_iter = 1 ):
    st.subheader("Hyperparameter Tuning Configuration")
   

    if tuning_method == "Random Search":
        search = RandomizedSearchCV(model, param_distributions=param_grid, cv=cv, n_iter=n_iter)

    elif tuning_method == "Grid Search":
        search = GridSearchCV(model, param_grid=param_grid, cv=cv)

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    st.write("Best Parameters:", search.best_params_)
    return best_model
