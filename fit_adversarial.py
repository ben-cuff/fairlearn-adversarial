import pandas as pd
import numpy as np

from fairlearn.adversarial import AdversarialFairnessRegressor, AdversarialFairnessClassifier
from fairlearn.metrics import MetricFrame

from fairness_measures import *

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    roc_auc_score,
    roc_curve,
    auc,
)


import dill

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, ReLU, Add
from tensorflow.keras import regularizers, optimizers, Input, Model

from tensorflow import keras

import matplotlib.pyplot as plt


def build_predictor(use_skip_connections=False, input_dim=None, is_regression=True):
    if not use_skip_connections and is_regression:
        return Sequential(
            [
                Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
                BatchNormalization(),
                LeakyReLU(),
                Dropout(0.3),
                Dense(128),
                LeakyReLU(),
                Dense(64),
                LeakyReLU(),
                Dense(32),
                LeakyReLU(),
                Dense(1, activation="linear"),
            ]
        )
    elif not use_skip_connections and not is_regression:
        return Sequential(
            [
                Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
                BatchNormalization(),
                LeakyReLU(),
                Dropout(0.3),
                Dense(128),
                LeakyReLU(),
                Dense(64),
                LeakyReLU(),
                Dense(32),
                LeakyReLU(),
                Dense(1, activation="sigmoid"),
            ]
        )

    if use_skip_connections and not is_regression:
        inputs = Input(shape=(input_dim,))
        x = Dense(512, kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x1 = Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU()(x1)
        x1 = Dropout(0.3)(x1)

        x_proj = Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
        x_skip = Add()([x_proj, x1])

        x2 = Dense(128)(x_skip)
        x2 = LeakyReLU()(x2)
        x3 = Dense(64)(x2)
        x3 = LeakyReLU()(x3)
        x4 = Dense(32)(x3)
        x4 = LeakyReLU()(x4)

        outputs = Dense(1, activation="sigmoid")(x4)

    inputs = Input(shape=(input_dim,))
    x = Dense(512, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x1 = Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    x1 = Dropout(0.3)(x1)

    # Project x to 256 to match x1
    x_proj = Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x_skip = Add()([x_proj, x1])

    x2 = Dense(128)(x_skip)
    x2 = LeakyReLU()(x2)
    x3 = Dense(64)(x2)
    x3 = LeakyReLU()(x3)
    x4 = Dense(32)(x3)
    x4 = LeakyReLU()(x4)

    outputs = Dense(1, activation="linear")(x4)
    return Model(inputs, outputs)


def build_adversary(use_skip_connections=False):
    if not use_skip_connections:
        return Sequential(
            [
                Dense(64, kernel_regularizer=regularizers.l2(1e-4)),
                BatchNormalization(),
                LeakyReLU(),
                Dense(32),
                ReLU(),
                Dense(16),
                ReLU(),
                Dense(1, activation="sigmoid"),
            ]
        )

    adv_inputs = Input(shape=(1,))
    a = Dense(128, kernel_regularizer=regularizers.l2(1e-4))(adv_inputs)
    a = BatchNormalization()(a)
    a = LeakyReLU()(a)

    a1 = Dense(64, kernel_regularizer=regularizers.l2(1e-4))(a)
    a1 = BatchNormalization()(a1)
    a1 = LeakyReLU()(a1)

    a_proj = Dense(64, kernel_regularizer=regularizers.l2(1e-4))(a)
    a_skip = Add()([a_proj, a1])

    a2 = Dense(32)(a_skip)
    a2 = ReLU()(a2)
    a3 = Dense(16)(a2)
    a3 = ReLU()(a3)
    adv_outputs = Dense(1, activation="sigmoid")(a3)
    return Model(adv_inputs, adv_outputs)


def fit_adversarial_regressor(
    alpha,
    sensitive_train,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    sensitive_val=None,
    is_regression=True,
    filename=None,
    batch_size=256,
    epochs=50,
    progress_updates=100,
    random_state=42,
    use_skip_connections=False,
    alpha_decay=None,
    learning_rate=0.001,
    learning_rate_decay=None,
    callbacks=None,
    update_alpha=None,
    update_epoch=None,
):
    predictor_model = build_predictor(
        use_skip_connections, input_dim=X_train.shape[1], is_regression=is_regression
    )
    adversary_model = build_adversary(use_skip_connections)

    if is_regression:
        mitigator = AdversarialFairnessRegressor(
            predictor_model=predictor_model,
            adversary_model=adversary_model,
            random_state=random_state,
            alpha=alpha,
            batch_size=batch_size,
            epochs=epochs,
            progress_updates=progress_updates,
            alpha_decay=alpha_decay,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            callbacks=callbacks,
        )
    else:
        mitigator = AdversarialFairnessClassifier(
            predictor_model=predictor_model,
            adversary_model=adversary_model,
            random_state=random_state,
            alpha=alpha,
            batch_size=batch_size,
            epochs=epochs,
            progress_updates=progress_updates,
            alpha_decay=alpha_decay,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            callbacks=callbacks,
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    mitigator.fit(
        X_train_scaled,
        y_train,
        sensitive_features=sensitive_train,
        X_val=X_val_scaled,
        y_val=y_val,
        sensitive_features_val=sensitive_val,
        update_alpha=update_alpha,
        update_epoch=update_epoch,
    )

    return mitigator, scaler


def test_model(
    mitigator, scaler, X_test, y_test, sensitive_features_test, verbose=False, is_regression=True
):
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    y_test = np.asarray(y_test).ravel()
    sensitive_features_test = np.asarray(sensitive_features_test).ravel()

    batch_size = 1000

    def _batched_predict(predict_fn):
        """Run predictions in batches to avoid exhausting GPU/CPU memory."""
        preds = []
        for i in range(0, len(X_test_scaled), batch_size):
            batch = X_test_scaled[i : i + batch_size]
            batch_pred = predict_fn(batch)
            preds.append(np.asarray(batch_pred).reshape(-1))
        return np.concatenate(preds) if preds else np.array([])

    if is_regression:
        y_pred = _batched_predict(mitigator.predict)

        if verbose:
            print(f"Predictions: {y_pred[:5]}")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if verbose:
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {rmse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")

        fairness_metrics = evaluate_fairness(y_pred, y_test, sensitive_features_test, verbose)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "fairness_metrics": fairness_metrics,
        }

    # Classification branch
    y_pred_proba = _batched_predict(mitigator._raw_predict).clip(0.0, 1.0)
    y_pred_labels = (y_pred_proba >= 0.5).astype(int)

    if verbose:
        print(f"Prediction probabilities: {y_pred_proba[:5]}")
        print(f"Prediction labels: {y_pred_labels[:5]}")

    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred_labels)
    report = classification_report(y_test, y_pred_labels)

    roc_auc = None
    roc_curve_data = None
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_curve_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": auc(fpr, tpr),
        }

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(report)

    fairness_metrics = evaluate_fairness(y_pred_proba, y_test, sensitive_features_test, verbose)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "roc_curve": roc_curve_data,
        "fairness_metrics": fairness_metrics,
    }


def evaluate_fairness(y_pred, target, minority, verbose=False):
    fairness_metrics = FairnessMetrics(y_pred, target, minority)
    metrics = fairness_metrics.calculate_metrics_with_cross_validation()
    fairness_metrics.train_classifiers(split_data=False)

    if verbose:
        print(metrics)
        for criterion in ["independence", "separation", "sufficiency"]:
            fairness_metrics.plot_roc_auc_curve(criterion)

    return metrics
