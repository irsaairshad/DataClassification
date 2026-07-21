"""
Project 2: Data Classification Using AI
DecodeLabs - AI Engineering Internship (Batch 2026)

Goal:
    Build a basic classification model using a small dataset (the Iris
    benchmark) and prove the full supervised-learning pipeline: load data,
    split into train/test, scale features, train a K-Nearest Neighbors
    classifier, and validate it with a confusion matrix and F1 score.

This follows the exact "Master Blueprint: IPO Framework" from the
Industrial Training Kit slides:
    INPUT   -> Iris Domain, Feature Scaling
    PROCESS -> Train-Test Split, KNN Algorithm
    OUTPUT  -> Confusion Matrix, F1 Score

Key Requirements met:
    - Load and understand a dataset            -> load_dataset()
    - Split data into training and testing sets -> train_test_split()
    - Apply a simple classification algorithm   -> KNeighborsClassifier

Author: Irsa (FA23-BCS-111), COMSATS University Islamabad - Vehari Campus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# PHASE 1 (INPUT): Raw Material - The Iris Benchmark
# 150 samples, 3 classes (Setosa, Versicolor, Virginica), 4 features
# (sepal length, sepal width, petal length, petal width).
# ---------------------------------------------------------------------------
def load_dataset():
    """Load and understand the Iris dataset (Key Requirement #1)."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    print("=" * 60)
    print("STEP 1: LOAD & UNDERSTAND THE DATASET")
    print("=" * 60)
    print(f"Samples: {df.shape[0]}  |  Features: {iris.data.shape[1]}  "
          f"|  Classes: {len(iris.target_names)}")
    print(f"Class names: {list(iris.target_names)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass balance (samples per species):")
    print(df["species"].value_counts())
    print()
    return iris, df


# ---------------------------------------------------------------------------
# PHASE 2 (PROCESS - part A): Structural Integrity - The Split
# Randomize (shuffle) before splitting to remove order bias, then hold out
# 20% of the data for testing (never seen during training).
# ---------------------------------------------------------------------------
def split_data(iris):
    """Split data into training and testing sets (Key Requirement #2)."""
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,     # 80% train / 20% test, per the IPO diagram
        random_state=42,    # reproducibility
        shuffle=True,       # "Shuffle" step from the Structural Integrity slide
        stratify=y,         # keep class balance equal in both sets
    )

    print("=" * 60)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}  |  Testing samples: {len(X_test)}")
    print()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# PHASE 2 (PROCESS - part B): The Gatekeeper Rule - Scaling
# StandardScaler transforms raw, biased-scale data into balanced data
# with mean = 0 and variance = 1, so no single feature dominates distance
# calculations in KNN.
# ---------------------------------------------------------------------------
def scale_features(X_train, X_test):
    """Feature scaling (Input stage of the IPO framework)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit only on train data
    X_test_scaled = scaler.transform(X_test)          # transform test with same scaler

    print("=" * 60)
    print("STEP 3: FEATURE SCALING (StandardScaler: mean=0, variance=1)")
    print("=" * 60)
    print("Scaling complete - no feature will dominate distance calculations.\n")
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# PHASE 3 (PROCESS): The Algorithm - K-Nearest Neighbors
# The Proximity Principle: similar things exist in close proximity.
# Workflow: INSTANTIATE -> FIT (memorize the map) -> PREDICT (apply logic)
# ---------------------------------------------------------------------------
def train_model(X_train_scaled, y_train, k=5):
    """Apply a simple classification algorithm (Key Requirement #3)."""
    print("=" * 60)
    print(f"STEP 4: TRAIN THE MODEL (K-Nearest Neighbors, k={k})")
    print("=" * 60)

    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train_scaled, y_train)             # FIT (memorize the map)
    print("Model trained (fitted) on the training set.\n")
    return model


# ---------------------------------------------------------------------------
# PHASE 4 (OUTPUT): Validation - Confusion Matrix & F1 Score
# "In imbalanced data, accuracy is a lie. We must look deeper" - hence we
# report the full confusion matrix, precision/recall, and F1 (harmonic mean
# of precision and recall) rather than accuracy alone.
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test_scaled, y_test, target_names):
    """Validate the model: confusion matrix + F1 score."""
    predictions = model.predict(X_test_scaled)   # PREDICT (apply logic)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print("=" * 60)
    print("STEP 5: OUTPUT VALIDATION")
    print("=" * 60)
    print(f"Accuracy: {acc:.2%}")
    print(f"Macro F1 Score: {f1:.2%}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(cm)
    print("\nFull Classification Report (precision, recall, F1 per class):")
    print(classification_report(y_test, predictions, target_names=target_names))

    return predictions, cm, acc, f1


def find_best_k(X_train_scaled, y_train, k_range=range(1, 21)):
    """Bonus (per 'Tuning the Engine' slide): sweep K to find the elbow
    (the K value with the lowest error rate).

    IMPORTANT FIX: K must be tuned using cross-validation on the TRAINING
    set only. Using the test set to choose K is data leakage - it lets
    the test set influence model selection, silently inflating the final
    reported score and defeating the purpose of a held-out test set.
    (This also explains why the old version picked K=1, which the
    "Tuning the Engine" slide itself flags as the overfitting/noise case.)
    """
    from sklearn.model_selection import cross_val_score

    mean_errors = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="accuracy")
        mean_errors.append(1 - scores.mean())

    best_k = list(k_range)[int(np.argmin(mean_errors))]

    plt.figure(figsize=(7, 5))
    plt.plot(list(k_range), mean_errors, marker="o", color="#1f4e79")
    plt.title("Tuning the Engine: Choosing K (5-Fold CV Error vs K)")
    plt.xlabel("K Value")
    plt.ylabel("Cross-Validated Error Rate")
    plt.axvline(best_k, color="orange", linestyle="--", label=f"Best K = {best_k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("k_tuning_curve.png", dpi=150)
    plt.close()

    print("=" * 60)
    print("BONUS: TUNING K (Elbow Method via Cross-Validation)")
    print("=" * 60)
    print(f"Best K found: {best_k} (lowest CV error = {min(mean_errors):.2%})")
    print("Saved plot -> k_tuning_curve.png\n")
    return best_k


def plot_confusion_matrix(cm, target_names):
    """Save a visual confusion matrix (Diagnostic Tool slide)."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Iris Classification (KNN)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved plot -> confusion_matrix.png\n")


def main():
    # STEP 1: INPUT - load & understand data
    iris, df = load_dataset()

    # STEP 2: PROCESS - split into train/test
    X_train, X_test, y_train, y_test = split_data(iris)

    # STEP 3: PROCESS - scale features (Gatekeeper Rule)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # BONUS: find the optimal K before final training (uses CV on train set only)
    best_k = find_best_k(X_train_scaled, y_train)

    # STEP 4: PROCESS - train the KNN model with the tuned K
    model = train_model(X_train_scaled, y_train, k=best_k)

    # STEP 5: OUTPUT - validate with confusion matrix & F1 score
    predictions, cm, acc, f1 = evaluate_model(
        model, X_test_scaled, y_test, iris.target_names
    )
    plot_confusion_matrix(cm, iris.target_names)

    print("=" * 60)
    print("PROJECT 2 COMPLETE")
    print("=" * 60)
    print(f"Final Model: KNN (k={best_k})  |  Accuracy: {acc:.2%}  |  F1: {f1:.2%}")


if __name__ == "__main__":
    main()
