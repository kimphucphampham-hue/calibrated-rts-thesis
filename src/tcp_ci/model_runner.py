from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


FORBIDDEN_FEATURES = {
    "failed",
    "execution_time_ms",
    "verdict_raw",
    "duration_raw",
    "split",
}


def validate_feature_list(
    features: list[str],
) -> None:
    overlap = (
        set(features)
        & FORBIDDEN_FEATURES
    )

    if overlap:
        raise ValueError(
            "Forbidden post-execution/model-control "
            "fields found in feature list: "
            + ", ".join(
                sorted(overlap)
            )
        )

    if len(features) != len(
        set(features)
    ):
        raise ValueError(
            "Duplicate feature names found."
        )


def build_numeric_preprocessor(
    feature_columns: list[str],
    preprocessing_config: dict,
) -> ColumnTransformer:
    validate_feature_list(
        feature_columns
    )

    imputer_config = (
        preprocessing_config[
            "imputer"
        ]
    )

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=imputer_config[
                    "strategy"
                ],
                add_indicator=bool(
                    imputer_config[
                        "add_indicator"
                    ]
                ),
            ),
        ),
    ]

    scaler_config = (
        preprocessing_config.get(
            "scaler",
            {},
        )
    )

    if bool(
        scaler_config.get(
            "enabled",
            False,
        )
    ):
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_columns,
            ),
        ],
        remainder="drop",
    )


def build_logistic_estimator(
    model_config: dict,
) -> LogisticRegression:
    return LogisticRegression(
        solver=model_config[
            "solver"
        ],
        C=float(
            model_config["C"]
        ),
        max_iter=int(
            model_config[
                "max_iter"
            ]
        ),
        class_weight=model_config[
            "class_weight"
        ],
    )



def build_random_forest_estimator(
    model_config: dict,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(
            model_config[
                "n_estimators"
            ]
        ),
        criterion=str(
            model_config[
                "criterion"
            ]
        ),
        max_depth=(
            None
            if model_config[
                "max_depth"
            ] is None
            else int(
                model_config[
                    "max_depth"
                ]
            )
        ),
        min_samples_split=int(
            model_config[
                "min_samples_split"
            ]
        ),
        min_samples_leaf=int(
            model_config[
                "min_samples_leaf"
            ]
        ),
        max_features=model_config[
            "max_features"
        ],
        bootstrap=bool(
            model_config[
                "bootstrap"
            ]
        ),
        class_weight=model_config[
            "class_weight"
        ],
        random_state=int(
            model_config[
                "random_state"
            ]
        ),
        n_jobs=int(
            model_config[
                "n_jobs"
            ]
        ),
    )

def build_xgboost_estimator(
    model_config: dict,
) -> XGBClassifier:
    return XGBClassifier(
        objective=str(
            model_config[
                "objective"
            ]
        ),
        booster=str(
            model_config[
                "booster"
            ]
        ),
        n_estimators=int(
            model_config[
                "n_estimators"
            ]
        ),
        learning_rate=float(
            model_config[
                "learning_rate"
            ]
        ),
        max_depth=int(
            model_config[
                "max_depth"
            ]
        ),
        min_child_weight=float(
            model_config[
                "min_child_weight"
            ]
        ),
        gamma=float(
            model_config[
                "gamma"
            ]
        ),
        subsample=float(
            model_config[
                "subsample"
            ]
        ),
        colsample_bytree=float(
            model_config[
                "colsample_bytree"
            ]
        ),
        reg_alpha=float(
            model_config[
                "reg_alpha"
            ]
        ),
        reg_lambda=float(
            model_config[
                "reg_lambda"
            ]
        ),
        scale_pos_weight=float(
            model_config[
                "scale_pos_weight"
            ]
        ),
        tree_method=str(
            model_config[
                "tree_method"
            ]
        ),
        random_state=int(
            model_config[
                "random_state"
            ]
        ),
        n_jobs=int(
            model_config[
                "n_jobs"
            ]
        ),
        eval_metric=str(
            model_config[
                "eval_metric"
            ]
        ),
    )


def build_probability_pipeline(
    feature_columns: list[str],
    preprocessing_config: dict,
    estimator,
) -> Pipeline:
    transformer = (
        build_numeric_preprocessor(
            feature_columns,
            preprocessing_config,
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                transformer,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def build_logistic_pipeline(
    feature_columns: list[str],
    config: dict,
) -> Pipeline:
    estimator = (
        build_logistic_estimator(
            config[
                "models"
            ][
                "logistic_regression"
            ]
        )
    )

    return build_probability_pipeline(
        feature_columns=feature_columns,
        preprocessing_config=config[
            "preprocessing"
        ],
        estimator=estimator,
    )



def build_random_forest_pipeline(
    feature_columns: list[str],
    config: dict,
) -> Pipeline:
    estimator = (
        build_random_forest_estimator(
            config[
                "models"
            ][
                "random_forest"
            ]
        )
    )

    return build_probability_pipeline(
        feature_columns=feature_columns,
        preprocessing_config=config[
            "preprocessing"
        ],
        estimator=estimator,
    )

def build_xgboost_pipeline(
    feature_columns: list[str],
    config: dict,
) -> Pipeline:
    estimator = (
        build_xgboost_estimator(
            config[
                "models"
            ][
                "xgboost"
            ]
        )
    )

    return build_probability_pipeline(
        feature_columns=feature_columns,
        preprocessing_config=config[
            "preprocessing"
        ],
        estimator=estimator,
    )


def fit_probability_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train length mismatch."
        )

    if len(X_train) == 0:
        raise ValueError(
            "Training data is empty."
        )

    classes = set(
        np.asarray(
            y_train,
            dtype=int,
        ).tolist()
    )

    if classes != {0, 1}:
        raise ValueError(
            "Training data must contain "
            "both target classes."
        )

    model.fit(
        X_train,
        y_train,
    )

    return model


def positive_probability(
    model,
    dataframe: pd.DataFrame,
) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(
            dataframe
        ),
        dtype=float,
    )

    if hasattr(
        model,
        "classes_",
    ):
        classes = list(
            model.classes_
        )
    else:
        classes = list(
            model[
                "model"
            ].classes_
        )

    if 1 not in classes:
        raise ValueError(
            "Positive class 1 not found."
        )

    positive_index = (
        classes.index(1)
    )

    result = probabilities[
        :,
        positive_index,
    ]

    if not np.isfinite(
        result
    ).all():
        raise ValueError(
            "Non-finite probability found."
        )

    if not (
        (result >= 0.0)
        &
        (result <= 1.0)
    ).all():
        raise ValueError(
            "Probability outside [0, 1]."
        )

    return result
