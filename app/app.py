import pandas as pd
from scipy import stats
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import shap
import joblib
import os
import requests
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
    text,
)
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
MLFLOW_DATABASE_PATH = PROJECT_ROOT / "experiments" / "mlflow.db"
MLFLOW_ARTIFACT_PATH = PROJECT_ROOT / "experiments" / "mlartifacts"

MLFLOW_ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)

LOCAL_MLFLOW_DATABASE_URI = (
    f"sqlite:///{MLFLOW_DATABASE_PATH.resolve().as_posix()}"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    LOCAL_MLFLOW_DATABASE_URI,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment_name = "Data Science Workbench"

existing_experiment = mlflow.get_experiment_by_name(experiment_name)

if existing_experiment is None:
    if MLFLOW_TRACKING_URI.startswith("http"):
        mlflow.create_experiment(experiment_name)
    else:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=MLFLOW_ARTIFACT_PATH.resolve().as_uri(),
        )

mlflow.set_experiment(experiment_name)

from src.database import (
    get_saved_datasets,
    run_select_query,
    save_dataset,
)

st.set_page_config(
    page_title="Data Science Workbench",
    page_icon="🔬",
    layout="wide",
)


def load_dataset(file):
    """Load an uploaded file into a Pandas DataFrame."""
    extension = file.name.rsplit(".", maxsplit=1)[-1].lower()

    if extension == "csv":
        return pd.read_csv(file)

    if extension in ["xlsx", "xls"]:
        return pd.read_excel(file)

    if extension == "json":
        return pd.read_json(file)

    raise ValueError("Unsupported file format.")

def show_data_cleaning_studio():
    st.header("Data Cleaning Studio")
    st.write("Clean your dataset while preserving the original uploaded file.")

    original_df = st.session_state.dataset

    if original_df is None:
        st.info(
            "Upload a dataset in Dataset Manager before using Data Cleaning Studio."
        )
        return

    if st.session_state.cleaned_dataset is None:
        st.session_state.cleaned_dataset = original_df.copy()

    df = st.session_state.cleaned_dataset.copy()

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Rows", f"{df.shape[0]:,}")
    metric_2.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    metric_3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Missing Value Handling")

    selected_column = st.selectbox(
        "Select a column to clean",
        df.columns.tolist(),
    )

    if pd.api.types.is_numeric_dtype(df[selected_column]):
        imputation_method = st.selectbox(
            "Choose an imputation method",
            ["Mean", "Median", "Mode"],
        )
    else:
        imputation_method = st.selectbox(
            "Choose an imputation method",
            ["Mode"],
        )

    if st.button("Apply missing-value treatment"):
        if df[selected_column].isna().sum() == 0:
            st.warning("This column has no missing values.")
        else:
            if imputation_method == "Mean":
                fill_value = df[selected_column].mean()

            elif imputation_method == "Median":
                fill_value = df[selected_column].median()

            else:
                mode_values = df[selected_column].mode(dropna=True)

                if mode_values.empty:
                    st.warning("A mode could not be calculated for this column.")
                    return

                fill_value = mode_values.iloc[0]

            df[selected_column] = df[selected_column].fillna(fill_value)
            st.session_state.cleaned_dataset = df
            st.success(f"Missing values filled using {imputation_method}.")
            st.rerun()

    st.subheader("Duplicate Removal")

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        st.warning(f"{duplicate_count:,} duplicate row(s) detected.")

        if st.button("Remove duplicate rows"):
            rows_before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            rows_removed = rows_before - len(df)

            st.session_state.cleaned_dataset = df
            st.success(f"Removed {rows_removed:,} duplicate row(s).")
            st.rerun()
    else:
        st.success("No duplicate rows found.")

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    cleaned_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download cleaned dataset as CSV",
        data=cleaned_csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

    if st.button("Reset all cleaning changes"):
        st.session_state.cleaned_dataset = original_df.copy()
        st.success("Cleaning changes have been reset.")
        st.rerun()

def show_dataset_manager():
    st.header("Dataset Manager")
    st.write("Upload a dataset and review its initial data-quality report.")

    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls", "json"],
        help="Supported formats: CSV, Excel, and JSON.",
    )

    if uploaded_file is not None:
        dataset_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if dataset_id != st.session_state.dataset_id:
            try:
                with st.spinner("Reading your dataset..."):
                    st.session_state.dataset = load_dataset(uploaded_file)
                    st.session_state.cleaned_dataset = (
                        st.session_state.dataset.copy()
                    )
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.dataset_id = dataset_id
                    st.session_state.saved_dataset_id = None
                    st.session_state.feature_dataset = None
                    st.session_state.model_result = None

                st.success(f"Loaded **{uploaded_file.name}** successfully.")

            except Exception as error:
                st.error(f"Could not read this dataset: {error}")
                return

    df = st.session_state.dataset

    if df is None:
        st.info("Upload a CSV, Excel, or JSON dataset to begin.")
        return

    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Rows", f"{df.shape[0]:,}")
    metric_2.metric("Columns", f"{df.shape[1]:,}")
    metric_3.metric("Missing values", f"{total_missing:,}")
    metric_4.metric("Duplicate rows", f"{duplicate_rows:,}")
    
    st.subheader("PostgreSQL Dataset Storage")

    if st.session_state.saved_dataset_id is None:
        if st.button(
            "Save Current Dataset to PostgreSQL",
            type="primary",
        ):
            try:
                with st.spinner("Saving dataset to PostgreSQL..."):
                    saved_id = save_dataset(
                        st.session_state.dataset_name,
                        df,
                    )

                st.session_state.saved_dataset_id = saved_id

                st.success(
                    f"Dataset saved successfully with database ID: {saved_id}"
                )

            except Exception as error:
                st.error(f"Could not save dataset to PostgreSQL: {error}")

    else:
        st.success(
            f"This uploaded dataset is saved in PostgreSQL with "
            f"database ID: {st.session_state.saved_dataset_id}."
        )

    if st.button("Show Saved Dataset History"):
        try:
            saved_datasets = get_saved_datasets()

            if saved_datasets:
                st.dataframe(
                    pd.DataFrame(saved_datasets),
                    use_container_width=True,
                )
            else:
                st.info("No datasets have been saved yet.")

        except Exception as error:
            st.error(f"Could not load saved datasets: {error}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    metadata = pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": df.dtypes.astype(str).values,
            "Missing values": df.isna().sum().values,
            "Unique values": df.nunique(dropna=False).values,
        }
    )

    st.subheader("Column Metadata")
    st.dataframe(metadata, use_container_width=True)

    st.subheader("Dataset Summary")
    st.write(f"**Current dataset:** {st.session_state.dataset_name}")
    st.write(f"**Memory usage:** {memory_mb:.2f} MB")
    st.write(f"**Total cells:** {df.shape[0] * df.shape[1]:,}")

    if total_missing > 0:
        st.subheader("Missing-Value Report")
        missing_report = metadata[metadata["Missing values"] > 0]
        st.dataframe(missing_report, use_container_width=True)
    else:
        st.success("No missing values found.")

    st.subheader("Descriptive Statistics")

    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) > 0:
        numeric_summary = df[numeric_columns].describe().T
        st.dataframe(numeric_summary, use_container_width=True)
    else:
        st.warning("This dataset does not contain numeric columns.")

    st.subheader("Download Data-Quality Report")

    row_count = max(len(df), 1)
    quality_report = pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Data type": df.dtypes.astype(str).values,
            "Missing values": df.isna().sum().values,
            "Missing percentage": (
                df.isna().sum() / row_count * 100
            ).round(2).values,
            "Unique values": df.nunique(dropna=False).values,
        }
    )

    report_csv = quality_report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download quality report as CSV",
        data=report_csv,
        file_name="data_quality_report.csv",
        mime="text/csv",
    )


def show_eda_studio():
    st.header("EDA Studio")
    st.write("Create interactive charts to understand the uploaded dataset.")

    df = st.session_state.dataset

    if df is None:
        st.info("Upload a dataset in Dataset Manager before using EDA Studio.")
        return

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    st.success(f"Exploring: **{st.session_state.dataset_name}**")

    chart_type = st.selectbox(
        "Choose a visualization",
        [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Scatter Plot",
            "Correlation Heatmap",
        ],
    )

    if chart_type == "Histogram":
        if not numeric_columns:
            st.warning("A histogram needs at least one numeric column.")
            return

        selected_column = st.selectbox(
            "Select a numeric column",
            numeric_columns,
        )

        figure = px.histogram(
            df,
            x=selected_column,
            nbins=30,
            title=f"Distribution of {selected_column}",
            template="plotly_white",
        )
        st.plotly_chart(figure, use_container_width=True)

    elif chart_type == "Box Plot":
        if not numeric_columns:
            st.warning("A box plot needs at least one numeric column.")
            return

        selected_column = st.selectbox(
            "Select a numeric column",
            numeric_columns,
        )

        figure = px.box(
            df,
            y=selected_column,
            title=f"Box Plot of {selected_column}",
            template="plotly_white",
        )
        st.plotly_chart(figure, use_container_width=True)

    elif chart_type == "Bar Chart":
        if not categorical_columns:
            st.warning("A bar chart needs at least one categorical column.")
            return

        selected_column = st.selectbox(
            "Select a categorical column",
            categorical_columns,
        )

        counts = (
            df[selected_column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(20)
            .reset_index()
        )
        counts.columns = [selected_column, "Count"]

        figure = px.bar(
            counts,
            x=selected_column,
            y="Count",
            title=f"Top Categories in {selected_column}",
            template="plotly_white",
        )
        st.plotly_chart(figure, use_container_width=True)

    elif chart_type == "Scatter Plot":
        if len(numeric_columns) < 2:
            st.warning("A scatter plot needs at least two numeric columns.")
            return

        column_1, column_2 = st.columns(2)

        with column_1:
            x_column = st.selectbox("X-axis", numeric_columns)

        with column_2:
            y_column = st.selectbox(
                "Y-axis",
                numeric_columns,
                index=1,
            )

        figure = px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=f"{y_column} vs {x_column}",
            template="plotly_white",
        )
        st.plotly_chart(figure, use_container_width=True)

    elif chart_type == "Correlation Heatmap":
        if len(numeric_columns) < 2:
            st.warning(
                "A correlation heatmap needs at least two numeric columns."
            )
            return

        correlation_matrix = df[numeric_columns].corr()

        figure = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap",
            aspect="auto",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.subheader("Automatic Data Insights")

    if numeric_columns:
        insight_rows = []

        for column in numeric_columns:
            values = df[column].dropna()

            if values.empty:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            outlier_count = int(
                ((values < lower_limit) | (values > upper_limit)).sum()
            )

            skewness = values.skew()

            if abs(skewness) < 0.5:
                skewness_label = "Approximately symmetric"
            elif skewness > 0:
                skewness_label = "Right-skewed"
            else:
                skewness_label = "Left-skewed"

            insight_rows.append(
                {
                    "Column": column,
                    "Skewness": round(skewness, 3),
                    "Distribution": skewness_label,
                    "Possible outliers": outlier_count,
                }
            )

        insights_df = pd.DataFrame(insight_rows)
        st.dataframe(insights_df, use_container_width=True)

        if len(numeric_columns) >= 2:
            correlation_threshold = st.slider(
                "High-correlation threshold",
                min_value=0.50,
                max_value=0.95,
                value=0.80,
                step=0.05,
            )

            correlation_matrix = df[numeric_columns].corr().abs()
            correlated_pairs = []

            for index_1 in range(len(numeric_columns)):
                for index_2 in range(index_1 + 1, len(numeric_columns)):
                    correlation_value = correlation_matrix.iloc[
                        index_1,
                        index_2,
                    ]

                    if correlation_value >= correlation_threshold:
                        correlated_pairs.append(
                            {
                                "Feature 1": numeric_columns[index_1],
                                "Feature 2": numeric_columns[index_2],
                                "Correlation": round(correlation_value, 3),
                            }
                        )

            st.write("**Potential multicollinearity**")

            if correlated_pairs:
                st.dataframe(
                    pd.DataFrame(correlated_pairs),
                    use_container_width=True,
                )
            else:
                st.success(
                    "No highly correlated numeric feature pairs were found."
                )
                
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

def show_feature_engineering_lab():
    st.header("Feature Engineering Lab")
    st.write(
        "Create machine-learning-ready features from the cleaned dataset."
    )

    source_df = st.session_state.cleaned_dataset

    if source_df is None:
        st.info(
            "Upload and clean a dataset before using Feature Engineering Lab."
        )
        return

    if st.session_state.feature_dataset is None:
        st.session_state.feature_dataset = source_df.copy()

    df = st.session_state.feature_dataset.copy()

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Rows", f"{df.shape[0]:,}")
    metric_2.metric("Features", f"{df.shape[1]:,}")
    metric_3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

    transformation = st.selectbox(
        "Choose a feature-engineering operation",
        ["One-Hot Encoding", "Feature Scaling"],
    )

    if transformation == "One-Hot Encoding":
        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        if not categorical_columns:
            st.warning("No categorical columns are available for encoding.")
        else:
            selected_columns = st.multiselect(
                "Select categorical columns",
                categorical_columns,
            )

            if st.button("Apply one-hot encoding"):
                if not selected_columns:
                    st.warning("Select at least one categorical column.")
                else:
                    df = pd.get_dummies(
                        df,
                        columns=selected_columns,
                        drop_first=False,
                        dtype=int,
                    )

                    st.session_state.feature_dataset = df
                    st.success("One-hot encoding applied successfully.")
                    st.rerun()

    elif transformation == "Feature Scaling":
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if not numeric_columns:
            st.warning("No numeric columns are available for scaling.")
        else:
            selected_columns = st.multiselect(
                "Select numeric columns",
                numeric_columns,
            )

            scaling_method = st.selectbox(
                "Choose a scaling method",
                ["Standard Scaling", "Min-Max Scaling"],
            )

            if st.button("Apply feature scaling"):
                if not selected_columns:
                    st.warning("Select at least one numeric column.")

                elif df[selected_columns].isna().any().any():
                    st.warning(
                        "Fill missing values in Data Cleaning Studio before scaling."
                    )

                else:
                    if scaling_method == "Standard Scaling":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()

                    df[selected_columns] = scaler.fit_transform(
                        df[selected_columns]
                    )

                    st.session_state.feature_dataset = df
                    st.success(f"{scaling_method} applied successfully.")
                    st.rerun()

    st.subheader("Feature-Engineered Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    engineered_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download feature-engineered dataset as CSV",
        data=engineered_csv,
        file_name="feature_engineered_dataset.csv",
        mime="text/csv",
    )

    if st.button("Reset feature-engineering changes"):
        st.session_state.feature_dataset = source_df.copy()
        st.success("Feature-engineering changes have been reset.")
        st.rerun()

def show_machine_learning_studio():
    st.header("Machine Learning Studio")
    st.write("Train and compare machine-learning models on your dataset.")

    df = st.session_state.feature_dataset

    if df is None:
        st.info(
            "Prepare a dataset in Feature Engineering Lab before training a model."
        )
        return

    if len(df) < 10:
        st.warning("Use a dataset with at least 10 rows for model training.")
        return

    task_type = st.selectbox(
        "Select machine-learning task",
        ["Classification", "Regression"],
    )

    target_column = st.selectbox(
        "Select target column",
        df.columns.tolist(),
    )

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    available_features = [
        column for column in numeric_columns if column != target_column
    ]

    selected_features = st.multiselect(
        "Select numeric feature columns",
        available_features,
        default=available_features,
    )

    test_size = st.slider(
        "Test dataset size",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )

    if task_type == "Classification":
        model_name = st.selectbox(
            "Select model",
            ["Logistic Regression", "Random Forest Classifier"],
        )
    else:
        model_name = st.selectbox(
            "Select model",
            ["Linear Regression", "Random Forest Regressor"],
        )

    if st.button("Train Model"):
        if not selected_features:
            st.warning("Select at least one numeric feature.")

        elif df[selected_features].isna().any().any():
            st.warning(
                "Your selected features contain missing values. "
                "Clean them before training."
            )

        elif df[target_column].isna().any():
            st.warning("Your target column contains missing values.")

        elif task_type == "Classification" and df[target_column].nunique() < 2:
            st.warning(
                "Classification needs a target column with at least two classes."
            )

        elif (
            task_type == "Regression"
            and not pd.api.types.is_numeric_dtype(df[target_column])
        ):
            st.warning("Regression needs a numeric target column.")

        else:
            X = df[selected_features]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
            )

            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                )

            elif model_name == "Linear Regression":
                model = LinearRegression()

            else:
                model = RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                )

            try:
                with st.spinner("Training model..."):
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                if task_type == "Classification":
                    metrics = {
                        "Accuracy": accuracy_score(y_test, predictions),
                        "Precision": precision_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        ),
                        "Recall": recall_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        ),
                        "F1 Score": f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        ),
                    }

                else:
                    metrics = {
                        "R2 Score": r2_score(y_test, predictions),
                        "Mean Absolute Error": mean_absolute_error(
                            y_test,
                            predictions,
                        ),
                        "Root Mean Squared Error": mean_squared_error(
                            y_test,
                            predictions,
                        ) ** 0.5,
                    }
                mlflow_run_id = None
                try:
                    with mlflow.start_run(
                        run_name=f"{model_name} Training Run"
                    ) as run:
                        mlflow.log_params(
                            {
                                "model_name": model_name,
                                "task_type": task_type,
                                "target_column": target_column,
                                "selected_features": ", ".join(
                                    selected_features
                                ),
                                "train_rows": len(X_train),
                                "test_rows": len(X_test),
                                "test_size": test_size,
                            }
                        )

                        mlflow.log_metrics(
                            {
                                metric_name: float(metric_value)
                                for metric_name, metric_value
                                in metrics.items()
                            }
                        )

                        mlflow.set_tags(
                            {
                                "dataset_name": str(
                                    st.session_state.dataset_name
                                ),
                                "project": "Data Science Workbench",
                            }
                        )

                        mlflow.sklearn.log_model(
                            sk_model=model,
                            name="trained_model",
                            input_example=X_train.head(3),
                        )

                        mlflow_run_id = run.info.run_id

                except Exception as error:
                    st.warning(
                        f"Model trained, but MLflow tracking failed: {error}"
                    )
                saved_model_path = None

                try:
                    model_directory = PROJECT_ROOT / "models"
                    model_directory.mkdir(parents=True, exist_ok=True)

                    model_path = model_directory / "latest_model.joblib"

                    model_bundle = {
                        "model": model,
                        "features": selected_features,
                        "target": target_column,
                        "task_type": task_type,
                        "model_name": model_name,
                    }

                    joblib.dump(model_bundle, model_path)

                    saved_model_path = str(model_path)

                except Exception as error:
                    st.warning(
                        f"Model trained, but could not save API model: {error}"
                    )
                    
                st.session_state.model_result = {
                    "task_type": task_type,
                    "model_name": model_name,
                    "model": model,
                    "features": selected_features,
                    "target": target_column,
                    "X_test": X_test,
                    "y_test": y_test,
                    "predictions": predictions,
                    "metrics": metrics,
                    "mlflow_run_id": mlflow_run_id,
                    "saved_model_path": saved_model_path,
                }

                st.success(f"{model_name} trained successfully.")

            except Exception as error:
                st.error(f"Model training failed: {error}")

    model_result = st.session_state.model_result

    if model_result is not None:
        st.subheader("Latest Model Result")
        st.write(f"**Task:** {model_result['task_type']}")
        st.write(f"**Model:** {model_result['model_name']}")
        st.write(f"**Target:** {model_result['target']}")

        metric_columns = st.columns(len(model_result["metrics"]))

        for column, (metric_name, metric_value) in zip(
            metric_columns,
            model_result["metrics"].items(),
        ):
            column.metric(metric_name, f"{metric_value:.3f}")

        st.subheader("Selected Features")
        st.write(", ".join(model_result["features"]))

def show_sql_analytics_workspace():
    st.header("SQL Analytics Workspace")
    st.write(
        "Run safe read-only SQL queries against datasets saved in PostgreSQL."
    )

    st.info(
        "Only SELECT queries are allowed. Dataset metadata is stored in "
        "`datasets` and uploaded rows are stored in `dataset_rows`."
    )

    example_queries = {
        "View saved dataset metadata": (
            "SELECT id, name, file_type, row_count, column_count, "
            "uploaded_at "
            "FROM datasets "
            "ORDER BY uploaded_at DESC;"
        ),
        "View stored dataset rows": (
            "SELECT dataset_id, row_number, row_data "
            "FROM dataset_rows "
            "ORDER BY dataset_id, row_number "
            "LIMIT 20;"
        ),
        "Count rows in every saved dataset": (
            "SELECT dataset_id, COUNT(*) AS stored_rows "
            "FROM dataset_rows "
            "GROUP BY dataset_id "
            "ORDER BY dataset_id;"
        ),
        "Write my own query": "SELECT * FROM datasets LIMIT 20;",
    }

    selected_example = st.selectbox(
        "Choose a query template",
        list(example_queries.keys()),
    )

    query = st.text_area(
        "SQL query",
        value=example_queries[selected_example],
        height=180,
    )

    if st.button("Run SQL Query", type="primary"):
        try:
            with st.spinner("Running SQL query..."):
                query_result = run_select_query(query)

            st.success(
                f"Query completed successfully. "
                f"{len(query_result):,} row(s) returned."
            )

            st.dataframe(query_result, use_container_width=True)

            if query not in st.session_state.sql_query_history:
                st.session_state.sql_query_history.insert(0, query)

        except Exception as error:
            st.error(f"Query could not be executed: {error}")

    if st.session_state.sql_query_history:
        with st.expander("Query History"):
            for number, previous_query in enumerate(
                st.session_state.sql_query_history,
                start=1,
            ):
                st.code(
                    f"{number}. {previous_query}",
                    language="sql",
                )

def show_prediction_center():
    st.header("Prediction Center")
    st.write(
        "Enter feature values and receive a prediction from the FastAPI model."
    )

    api_url = st.text_input(
        "FastAPI server URL",
        value="http://127.0.0.1:8000",
    ).rstrip("/")

    try:
        response = requests.get(
            f"{api_url}/model-info",
            timeout=5,
        )
        response.raise_for_status()

        model_info = response.json()

    except requests.RequestException:
        st.warning(
            "FastAPI is not available. Start it in another terminal using "
            "`uvicorn api.main:app --reload --port 8000`."
        )
        return

    st.success("Connected to the FastAPI prediction backend.")

    st.write(f"**Model:** {model_info['model_name']}")
    st.write(f"**Task:** {model_info['task_type']}")
    st.write(f"**Target:** {model_info['target_column']}")

    st.subheader("Enter Feature Values")

    input_values = {}
    required_features = model_info["required_features"]

    input_columns = st.columns(2)

    for index, feature in enumerate(required_features):
        with input_columns[index % 2]:
            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                key=f"prediction_input_{feature}",
            )

    if st.button("Generate Prediction", type="primary"):
        try:
            prediction_response = requests.post(
                f"{api_url}/predict",
                json={"features": input_values},
                timeout=15,
            )

            prediction_response.raise_for_status()

            prediction_result = prediction_response.json()

            st.subheader("Prediction Result")
            st.success(
                f"Predicted value: **{prediction_result['prediction']}**"
            )

            if "class_probabilities" in prediction_result:
                st.subheader("Class Probabilities")

                probability_df = pd.DataFrame(
                    prediction_result["class_probabilities"].items(),
                    columns=["Class", "Probability"],
                )

                probability_figure = px.bar(
                    probability_df,
                    x="Class",
                    y="Probability",
                    title="Prediction Probability by Class",
                    template="plotly_white",
                )

                st.plotly_chart(
                    probability_figure,
                    use_container_width=True,
                )

        except requests.RequestException as error:
            st.error(f"Prediction request failed: {error}")

def initialise_session_state():
    """Create values that must remain available while navigating pages."""

    if "dataset" not in st.session_state:
        st.session_state.dataset = None

    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None

    if "cleaned_dataset" not in st.session_state:
        st.session_state.cleaned_dataset = None

    if "dataset_id" not in st.session_state:
        st.session_state.dataset_id = None

    if "feature_dataset" not in st.session_state:
        st.session_state.feature_dataset = None

    if "model_result" not in st.session_state:
        st.session_state.model_result = None

    if "saved_dataset_id" not in st.session_state:
        st.session_state.saved_dataset_id = None
    
    if "sql_query_history" not in st.session_state:
        st.session_state.sql_query_history = []

def show_model_evaluation_center():
    st.header("Model Evaluation Center")
    st.write("Review the performance of the most recently trained model.")

    model_result = st.session_state.model_result

    if model_result is None:
        st.info(
            "Train a model in Machine Learning Studio before viewing evaluation."
        )
        return

    task_type = model_result["task_type"]
    y_test = model_result["y_test"]
    predictions = model_result["predictions"]

    st.write(f"**Model:** {model_result['model_name']}")
    st.write(f"**Task:** {task_type}")
    st.write(f"**Target column:** {model_result['target']}")

    st.subheader("Performance Metrics")

    metric_columns = st.columns(len(model_result["metrics"]))

    for column, (metric_name, metric_value) in zip(
        metric_columns,
        model_result["metrics"].items(),
    ):
        column.metric(metric_name, f"{metric_value:.3f}")

    results_df = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
        }
    )

    if task_type == "Classification":
        st.subheader("Confusion Matrix")

        labels = sorted(
            set(y_test.astype(str)).union(set(pd.Series(predictions).astype(str)))
        )

        matrix = confusion_matrix(
            y_test.astype(str),
            pd.Series(predictions).astype(str),
            labels=labels,
        )

        figure = px.imshow(
            matrix,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
            labels={
                "x": "Predicted Label",
                "y": "Actual Label",
                "color": "Count",
            },
        )

        st.plotly_chart(figure, use_container_width=True)

        st.subheader("Actual vs Predicted Results")
        st.dataframe(results_df, use_container_width=True)

    else:
        st.subheader("Actual vs Predicted Plot")

        figure = px.scatter(
            results_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Values",
            template="plotly_white",
        )

        minimum_value = min(
            results_df["Actual"].min(),
            results_df["Predicted"].min(),
        )
        maximum_value = max(
            results_df["Actual"].max(),
            results_df["Predicted"].max(),
        )

        figure.add_shape(
            type="line",
            x0=minimum_value,
            y0=minimum_value,
            x1=maximum_value,
            y1=maximum_value,
            line={"color": "red", "dash": "dash"},
        )

        st.plotly_chart(figure, use_container_width=True)

        st.subheader("Residual Plot")

        results_df["Residual"] = (
            results_df["Actual"] - results_df["Predicted"]
        )

        residual_figure = px.scatter(
            results_df,
            x="Predicted",
            y="Residual",
            title="Residual Plot",
            template="plotly_white",
        )

        residual_figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
        )

        st.plotly_chart(residual_figure, use_container_width=True)

        st.subheader("Actual vs Predicted Results")
        st.dataframe(results_df, use_container_width=True)

    results_csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download prediction results as CSV",
        data=results_csv,
        file_name="model_prediction_results.csv",
        mime="text/csv",
    )

def show_statistical_analysis_lab():
    st.header("Statistical Analysis Lab")
    st.write("Run common statistical tests on the feature-engineered dataset.")

    df = st.session_state.feature_dataset

    if df is None:
        st.info(
            "Upload and prepare a dataset before using Statistical Analysis Lab."
        )
        return

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    test_name = st.selectbox(
        "Choose a statistical test",
        [
            "Pearson Correlation",
            "Spearman Correlation",
            "Independent T-Test",
            "One-Way ANOVA",
            "Chi-Square Test",
        ],
    )

    st.divider()

    if test_name in ["Pearson Correlation", "Spearman Correlation"]:
        if len(numeric_columns) < 2:
            st.warning("This test needs at least two numeric columns.")
            return

        column_1, column_2 = st.columns(2)

        with column_1:
            first_column = st.selectbox(
                "Select first numeric column",
                numeric_columns,
            )

        with column_2:
            second_column = st.selectbox(
                "Select second numeric column",
                numeric_columns,
                index=1,
            )

        if st.button("Run correlation test"):
            test_df = df[[first_column, second_column]].dropna()

            if len(test_df) < 3:
                st.warning("At least 3 valid rows are required.")
                return

            if test_name == "Pearson Correlation":
                statistic, p_value = stats.pearsonr(
                    test_df[first_column],
                    test_df[second_column],
                )
            else:
                statistic, p_value = stats.spearmanr(
                    test_df[first_column],
                    test_df[second_column],
                )

            result_1, result_2 = st.columns(2)
            result_1.metric("Correlation coefficient", f"{statistic:.3f}")
            result_2.metric("P-value", f"{p_value:.4f}")

            if abs(statistic) < 0.3:
                strength = "weak"
            elif abs(statistic) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"

            direction = "positive" if statistic >= 0 else "negative"

            if p_value < 0.05:
                conclusion = (
                    f"There is a statistically significant {strength} "
                    f"{direction} relationship between the two variables."
                )
            else:
                conclusion = (
                    "The observed relationship is not statistically "
                    "significant at the 0.05 level."
                )

            st.info(conclusion)

    elif test_name in ["Independent T-Test", "One-Way ANOVA"]:
        if not numeric_columns or not categorical_columns:
            st.warning(
                "This test needs at least one numeric and one categorical column."
            )
            return

        numeric_column = st.selectbox(
            "Select numeric value column",
            numeric_columns,
        )

        group_column = st.selectbox(
            "Select categorical grouping column",
            categorical_columns,
        )

        valid_groups = (
            df[group_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if test_name == "Independent T-Test":
            if len(valid_groups) < 2:
                st.warning("T-test needs at least two groups.")
                return

            group_1, group_2 = st.columns(2)

            with group_1:
                first_group = st.selectbox(
                    "Select first group",
                    valid_groups,
                )

            with group_2:
                second_group = st.selectbox(
                    "Select second group",
                    valid_groups,
                    index=1,
                )

            if st.button("Run T-test"):
                if first_group == second_group:
                    st.warning("Select two different groups.")
                    return

                first_values = df.loc[
                    df[group_column].astype(str) == first_group,
                    numeric_column,
                ].dropna()

                second_values = df.loc[
                    df[group_column].astype(str) == second_group,
                    numeric_column,
                ].dropna()

                if len(first_values) < 2 or len(second_values) < 2:
                    st.warning("Each group needs at least two valid values.")
                    return

                statistic, p_value = stats.ttest_ind(
                    first_values,
                    second_values,
                    equal_var=False,
                )

                result_1, result_2 = st.columns(2)
                result_1.metric("T statistic", f"{statistic:.3f}")
                result_2.metric("P-value", f"{p_value:.4f}")

                st.write(f"**{first_group} mean:** {first_values.mean():.3f}")
                st.write(f"**{second_group} mean:** {second_values.mean():.3f}")

                if p_value < 0.05:
                    st.success(
                        "The difference between the group means is "
                        "statistically significant."
                    )
                else:
                    st.info(
                        "The difference between the group means is not "
                        "statistically significant at the 0.05 level."
                    )

        else:
            selected_groups = st.multiselect(
                "Select two or more groups",
                valid_groups,
                default=valid_groups[: min(3, len(valid_groups))],
            )

            if st.button("Run ANOVA"):
                if len(selected_groups) < 2:
                    st.warning("Select at least two groups.")
                    return

                group_values = []

                for group in selected_groups:
                    values = df.loc[
                        df[group_column].astype(str) == group,
                        numeric_column,
                    ].dropna()

                    if len(values) < 2:
                        st.warning(
                            f"Group '{group}' needs at least two valid values."
                        )
                        return

                    group_values.append(values)

                statistic, p_value = stats.f_oneway(*group_values)

                result_1, result_2 = st.columns(2)
                result_1.metric("F statistic", f"{statistic:.3f}")
                result_2.metric("P-value", f"{p_value:.4f}")

                if p_value < 0.05:
                    st.success(
                        "At least one selected group has a statistically "
                        "different mean."
                    )
                else:
                    st.info(
                        "No statistically significant difference was found "
                        "among the selected group means."
                    )

    else:
        if len(categorical_columns) < 2:
            st.warning("Chi-square needs at least two categorical columns.")
            return

        column_1, column_2 = st.columns(2)

        with column_1:
            first_column = st.selectbox(
                "Select first categorical column",
                categorical_columns,
            )

        with column_2:
            second_column = st.selectbox(
                "Select second categorical column",
                categorical_columns,
                index=1,
            )

        if st.button("Run Chi-Square Test"):
            if first_column == second_column:
                st.warning("Select two different columns.")
                return

            contingency_table = pd.crosstab(
                df[first_column].fillna("Missing").astype(str),
                df[second_column].fillna("Missing").astype(str),
            )

            if contingency_table.empty:
                st.warning("A contingency table could not be created.")
                return

            statistic, p_value, degrees_of_freedom, _ = stats.chi2_contingency(
                contingency_table
            )

            result_1, result_2, result_3 = st.columns(3)
            result_1.metric("Chi-square statistic", f"{statistic:.3f}")
            result_2.metric("P-value", f"{p_value:.4f}")
            result_3.metric("Degrees of freedom", degrees_of_freedom)

            st.subheader("Contingency Table")
            st.dataframe(contingency_table, use_container_width=True)

            if p_value < 0.05:
                st.success(
                    "There is a statistically significant association "
                    "between the selected categorical variables."
                )
            else:
                st.info(
                    "No statistically significant association was found "
                    "at the 0.05 level."
                )

def show_explainability_studio():
    st.header("Explainability Studio")
    st.write(
        "Understand which input features have the greatest influence on "
        "the latest trained model."
    )

    model_result = st.session_state.model_result

    if model_result is None:
        st.info(
            "Train a model in Machine Learning Studio before using "
            "Explainability Studio."
        )
        return

    model = model_result["model"]
    features = model_result["features"]

    st.write(f"**Model:** {model_result['model_name']}")
    st.write(f"**Target:** {model_result['target']}")

    importance_values = None
    importance_method = None

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
        importance_method = "Model Feature Importance"

    elif hasattr(model, "coef_"):
        coefficients = model.coef_

        if len(coefficients.shape) == 1:
            importance_values = abs(coefficients)
        else:
            importance_values = abs(coefficients).mean(axis=0)

        importance_method = "Absolute Model Coefficient"

    if importance_values is None:
        st.warning(
            "This model does not provide built-in feature importance. "
            "Train a Random Forest or linear model to view this section."
        )
        return

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": importance_values,
        }
    ).sort_values("Importance", ascending=False)

    st.subheader("Feature Importance Ranking")
    st.dataframe(importance_df, use_container_width=True)

    figure = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=importance_method,
        template="plotly_white",
    )

    figure.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(figure, use_container_width=True)

    most_important = importance_df.iloc[0]

    st.success(
        f"The most influential feature is **{most_important['Feature']}** "
        f"with an importance score of "
        f"**{most_important['Importance']:.3f}**."
    )

    importance_csv = importance_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download feature importance as CSV",
        data=importance_csv,
        file_name="feature_importance.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("SHAP Global Explanation")

    if st.button("Generate SHAP Feature Importance"):
        try:
            X_test = model_result["X_test"]

            with st.spinner("Generating SHAP explanation..."):
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test)

            shap.plots.bar(
                shap_values,
                max_display=15,
                show=False,
            )

            st.pyplot(plt.gcf())
            plt.close()

            st.success(
                "SHAP explanation generated. Features at the top have the "
                "greatest overall influence on model predictions."
            )

        except Exception as error:
            st.error(f"Could not generate the SHAP explanation: {error}")

    st.divider()
    st.subheader("LIME Individual Prediction Explanation")

    if "X_test" not in model_result:
        st.warning("Train a new model before creating a LIME explanation.")
        return

    X_test = model_result["X_test"].reset_index(drop=True)

    selected_row = st.number_input(
        "Select a test-data row to explain",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1,
    )

    if st.button("Generate LIME Explanation"):
        try:
            if model_result["task_type"] == "Classification":
                explainer = LimeTabularExplainer(
                    training_data=X_test.to_numpy(),
                    feature_names=features,
                    class_names=[
                        str(value) for value in model.classes_
                    ],
                    mode="classification",
                )

                explanation = explainer.explain_instance(
                    X_test.iloc[selected_row].to_numpy(),
                    model.predict_proba,
                )

            else:
                explainer = LimeTabularExplainer(
                    training_data=X_test.to_numpy(),
                    feature_names=features,
                    mode="regression",
                )

                explanation = explainer.explain_instance(
                    X_test.iloc[selected_row].to_numpy(),
                    model.predict,
                )

            lime_df = pd.DataFrame(
                explanation.as_list(),
                columns=["Feature Condition", "Contribution"],
            )

            st.dataframe(lime_df, use_container_width=True)

            lime_figure = px.bar(
                lime_df,
                x="Contribution",
                y="Feature Condition",
                orientation="h",
                title=f"LIME Explanation for Test Row {selected_row}",
                template="plotly_white",
            )

            lime_figure.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(lime_figure, use_container_width=True)

        except Exception as error:
            st.error(f"Could not generate the LIME explanation: {error}")

def show_experiment_tracking():
    st.header("MLflow Experiment Tracking")
    st.write(
        "Compare saved machine-learning experiments, parameters, "
        "and performance metrics."
    )

    experiment = mlflow.get_experiment_by_name(
        "Data Science Workbench"
    )

    if experiment is None:
        st.info("No MLflow experiment has been created yet.")
        return

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )

    if runs_df.empty:
        st.info(
            "No training runs have been recorded yet. "
            "Train a model in Machine Learning Studio."
        )
        return

    display_columns = [
        column
        for column in runs_df.columns
        if column.startswith("metrics.")
        or column.startswith("params.")
        or column in [
            "run_id",
            "status",
            "start_time",
            "end_time",
        ]
    ]

    st.subheader("Recorded Training Runs")
    st.dataframe(
        runs_df[display_columns],
        use_container_width=True,
    )

    st.caption(
        "Start the MLflow interface from a separate terminal to inspect "
        "saved models and artifacts visually."
    )

    st.code(
        "mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db --port 5000",
        language="powershell",
    )

initialise_session_state()
st.sidebar.title("🔬 Workbench")
st.sidebar.caption("End-to-end Data Science Platform")

selected_page = st.sidebar.radio(
    "Navigate",
    [
    "Dataset Manager",
    "EDA Studio",
    "Data Cleaning Studio",
    "Feature Engineering Lab",
    "Machine Learning Studio",
    "Model Evaluation Center",
    "Statistical Analysis Lab",
    "Explainability Studio",
    "SQL Analytics Workspace",
    "Experiment Tracking",
    "Prediction Center",
    ],
)

st.sidebar.divider()
st.sidebar.caption("College Minor Project")
st.sidebar.caption("Built with Python and Streamlit")

st.title("Data Science Workbench")
st.caption("A unified platform for data analysis and machine learning.")

if selected_page == "Dataset Manager":
    show_dataset_manager()

if selected_page == "EDA Studio":
    show_eda_studio()
    
if selected_page == "Data Cleaning Studio":
    show_data_cleaning_studio()

if selected_page == "Feature Engineering Lab":
    show_feature_engineering_lab()

if selected_page == "Machine Learning Studio":
    show_machine_learning_studio()

if selected_page == "Model Evaluation Center":
    show_model_evaluation_center()
    
if selected_page == "Statistical Analysis Lab":
    show_statistical_analysis_lab()

if selected_page == "Explainability Studio":
    show_explainability_studio()

if selected_page == "SQL Analytics Workspace":
    show_sql_analytics_workspace()
    
if selected_page == "Experiment Tracking":
    show_experiment_tracking()

if selected_page == "Prediction Center":
    show_prediction_center()