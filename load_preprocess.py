"""
Variant Classification Pipeline with Hydra Configuration
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def apply_preprocessing(X: pd.DataFrame, cfg: DictConfig):
    """Applies a series of preprocessing steps to the feature set (X)."""

    features_to_drop = []
    if "drop_feature_selectors" in cfg.preprocessing:
        selectors = cfg.preprocessing.drop_feature_selectors
        selected_cols = set()

        if "regex_pattern" in selectors:
            pattern = selectors.regex_pattern
            matching_cols = X.filter(regex=pattern).columns
            selected_cols.update(matching_cols)

        if "slice_by_name" in selectors and len(selectors.slice_by_name) == 2:
            start_col, end_col = selectors.slice_by_name
            start_idx = X.columns.get_loc(start_col)
            end_idx = X.columns.get_loc(end_col)
            selected_cols.update(X.columns[start_idx : end_idx + 1])

        if "explicit_list" in selectors:
            selected_cols.update(selectors.explicit_list)

        features_to_drop = sorted(list(selected_cols))
    print(f"Dropping features {features_to_drop}")
    X = X.drop(columns=features_to_drop, errors="ignore")

    if cfg.preprocessing.get("select_numeric_features", False):
        X = X.select_dtypes(include=["number"])

    # Create preprocessing pipeline taken from conf/preprocessing
    pipeline_steps = []
    if "steps" in cfg.preprocessing and cfg.preprocessing.steps:
        for i, step_conf in enumerate(cfg.preprocessing.steps):
            # print(f"Step: Adding pipeline step {i+1}: {step_conf._target_}")
            step_instance = instantiate(step_conf)
            pipeline_steps.append(
                (f"step_{i}_{step_conf._target_.split('.')[-1]}", step_instance)
            )

    if pipeline_steps:
        pipeline = Pipeline(steps=pipeline_steps)
        print("\nApplying preprocessing pipeline...")
        column_names = X.columns
        X_processed = pipeline.fit_transform(X)
        try:
            new_columns = pipeline.get_feature_names_out(column_names)
        except AttributeError:
            new_columns = (
                column_names
                if not hasattr(pipeline.steps[-1][1], "get_feature_names_out")
                else pipeline.steps[-1][1].get_feature_names_out(column_names)
            )

        X = pd.DataFrame(X_processed, columns=new_columns, index=X.index)

    return X


def fit_scaler(X: pd.DataFrame, cfg: DictConfig):
    # Normalization/Scaling
    scaler_type = cfg.preprocessing.get("scaler", None)
    scaler = None
    if scaler_type:
        print(f"\n\n\n\nFitting {scaler_type} scaler...\n\n\n\n")

        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            print(f"Warning: Unknown scaler type '{scaler_type}', skipping scaling.")

        if scaler is not None:
            scaler.fit(X)
    return scaler


def load_and_preprocess_data(cfg: DictConfig):
    """Loads and preprocesses the dataset based on the config."""
    print(f"Loading dataset: {cfg.dataset.name} from {cfg.dataset.path}")

    # 1. Loading data
    try:
        data = pd.read_csv(cfg.dataset.path, low_memory=cfg.dataset.low_memory)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {data.shape}")

    # 2. Check if target column exists in the dataframe
    if cfg.dataset.target_column not in data.columns:
        print(
            f"Error: Target column '{cfg.dataset.target_column}' not found after preprocessing."
        )
        return None, None, None, None, None, None

    X = data.drop(columns=[cfg.dataset.target_column])
    # print(f"1. ---- {len(X.columns)}")
    y = data[cfg.dataset.target_column]

    # 3. Apply all preprocessing steps to the features X (taken from conf/preprocessing)
    print(f"\n--- Applying Preprocessing on Features: {cfg.preprocessing.name} ---")
    X_processed = apply_preprocessing(X, cfg)
    print("--- Preprocessing Complete ---\n")

    # 4. Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 5. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y_encoded,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y_encoded,
    )

    early_stopping_rounds = 0
    val_size = 0
    if "early_stopping_rounds" in cfg.model:
        early_stopping_rounds = cfg.model.early_stopping_rounds
        val_size = cfg.model.validation_size
        del cfg.model.validation_size

    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=cfg.training.random_state,
            stratify=y_train,
        )
    else:
        X_val, y_val = None, None

    fill_strategy = cfg.preprocessing.get("fill_na_strategy")
    if fill_strategy:
        print(f"Instantiating imputer with strategy '{fill_strategy}'")
        numeric_cols = X_train.select_dtypes(include=["number"]).columns
        imputer = SimpleImputer(strategy=fill_strategy)
        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        if X_val is not None:
            X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
    else:
        imputer = None

    scaler = fit_scaler(X_train, cfg)
    if scaler is not None:
        col_names = X_train.columns
        X_train_scaled = scaler.transform(X_train)
        X_train = pd.DataFrame(X_train_scaled, columns=col_names, index=X_train.index)
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
            X_val = pd.DataFrame(X_val_scaled, columns=col_names, index=X_train.index)
        X_test_scaled = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test_scaled, columns=col_names, index=X_test.index)

    return X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, imputer
