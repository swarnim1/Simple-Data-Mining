import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression , Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score ,mean_absolute_error , r2_score,precision_score,recall_score,f1_score
from components import utils
from components import evaluate_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier 



def train_model(data):
    st.header("Model Training")

    # Step 1: Target Variable Selection
    target_column = st.selectbox("Select Target Column", data.columns)
    target_dtype = data[target_column].dtype

    # Determine if target is continuous or categorical
    if np.issubdtype(target_dtype, np.number):
        is_continuous = True
        st.text("Target variable is continuous. Suggested models: Regression models")
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Polynomial Regression": None,  # We'll handle it differently
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "XGBoost Regressor": XGBRegressor(),
            "KNN Regressor": KNeighborsRegressor(),
        }
    else:
        is_continuous = False
        st.text("Target variable is categorical. Suggested models: Classification models")
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "KNN Classifier": KNeighborsClassifier(),
        }

    # Step 2: Feature Selection
    features = st.multiselect(
        "Select Features (excluding the target column)",
        [col for col in data.columns if col != target_column],
        default=[col for col in data.columns if col != target_column]
    )

    if len(features) == 0:
        st.warning("Please select at least one feature.")
        return

    X = data[features]
    y = data[target_column]

    # Preprocess categorical features in X
    if not X.select_dtypes(include=['number']).equals(X):
        X = pd.get_dummies(X, drop_first=True)

    # Handle categorical target variable for classification
    if not np.issubdtype(y.dtype, np.number) and not is_continuous:
        y = y.astype('category').cat.codes

    # Step 3: Train-Test Split
    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
    random_state = st.number_input("Random State", value=42, step=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step 4: Model Selection
    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    if model_name == "Polynomial Regression":
        poly_degree = st.slider("Select Polynomial Degree", 2, 5, 2)
        poly = PolynomialFeatures(degree=poly_degree)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        model = LinearRegression()  # Polynomial regression uses LinearRegression underneath

    # Step 5: Hyperparameter Tuning
    if(model_name != "Linear Regression" and model_name != "Polynomial Regression" ):

        tuning_method = st.radio("Select Tuning Method", ["None", "Grid Search", "Random Search"], index=0)
        param_grid = {}

        if tuning_method in ["Grid Search", "Random Search"]:
            param_grid = utils.configure_hyperparameters(model_name, tuning_method)
            cv = st.slider("Number of Cross-Validation Folds (cv)", 2, 10, 3)

            if tuning_method == "Random Search":
                n_iter = st.number_input("Number of Iterations (n_iter) for Random Search", 1, 100, 10)
                if st.button("Run Hyperparameter Search"):
                    model = utils.run_hyperparameter_search(model, param_grid, X_train, y_train, tuning_method, n_iter, cv)
            else:
                if st.button("Run Hyperparameter Search"):
                    model = utils.run_hyperparameter_search(model, param_grid, X_train, y_train, tuning_method, cv)

    # Step 6: Train and Test Button
    if st.button("Train and Test"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, X_test, y_test, y_pred, is_continuous
    return None, None, None, None, None

        



