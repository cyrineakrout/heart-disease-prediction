import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers import *
from implementations import *
from data_cleaning.pca import *
from cross_validation import *
from data_cleaning.filling_features import *
from data_cleaning.col_clean import *

x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    "dataset", sub_sample=False
)

import numpy as np


def impute_X_with_zero(X):
    """
    Impute missing values (NaN) in X with 0.

    Args:
        X (np.ndarray): Training data matrix.

    Returns:
        X (np.ndarray): Arrays with NaNs replaced by 0.
    """
    for idx, col in enumerate(X.T):
        # replace NaN with 0 in both train and test sets
        mask_train = np.isnan(X[:, idx])
        if np.any(mask_train):
            X[mask_train, idx] = 0
    return X


# ---------- Utility: evaluation ----------
def evaluate_f1(y, X):
    """Compute mean validation F1-score using 5-fold CV."""
    seed = np.random.seed(12)
    idx = np.random.RandomState(0).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    # apply reg_logistic_regression to find the weights
    weights, loss = reg_logistic_regression(
        ytr,
        Xtr,
        lambda_=0.001,
        initial_w=np.zeros(X.shape[1]),
        max_iters=2000,
        gamma=0.1,
    )
    probs_va = sigmoid(Xva @ weights)

    ths = np.linspace(0.02, 0.98, 49)
    best_f1, best_th = -1, 0.5
    for th in ths:
        yhat01 = (probs_va >= th).astype(int)
        yhat_pm1 = 2 * yhat01 - 1
        f1 = f1_score(2 * yva - 1, yhat_pm1)  # yva in {0,1} → map to {-1,+1}
        if f1 > best_f1:
            best_f1, best_th = f1, th

    f1s = f1_score(yva, np.where(sigmoid(Xva @ weights) >= best_th, 1, 0))
    return f1s


# ---------- Baseline preprocessing ----------
def baseline_without_pca(y, X):
    """Median imputation + standardization + bias (always applied)."""
    y = np.copy(y)
    y[y == -1] = 0

    # Replace NaN by median
    X = impute_X_with_zero(X)

    # Standardize
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    X = (X - mean) / std

    # Add bias
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return y, X


def baseline_with_pca(y, X):
    """Median imputation + standardization + bias (always applied)."""
    y = np.copy(y)
    y[y == -1] = 0

    # Replace NaN by median
    X = impute_X_with_zero(X)

    # Standardize
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    X = (X - mean) / std

    # apply PCA
    cat_cols = detect_categorical_cols(X)
    cont_cols = detect_continuous_cols(X)
    d_opt, _ = choose_d_by_variance(X[:, cont_cols], threshold=0.95)
    pca = PCA(d_opt)
    pca.find_principal_components(X[:, cont_cols])
    X = np.c_[X[:, cat_cols], pca.reduce_dimension(X[:, cont_cols])]

    # Add bias
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return y, X


# ---------- Stepwise additions ----------
def add_fill_features(y, X):
    X, _ = fill_all_important_features(X, X)
    return y, X, X


def add_remove_useless(y, X):
    X = np.c_[X[:, 0], X[:, 9:]]
    return y, X, X


def add_make_special_codes_nan(y, X):
    X, _ = make_special_codes_nan(X, X)
    return y, X, X


def add_drop_highnan(y, X):
    X, _ = drop_high_nan_columns(X, X)
    return y, X, X


def add_drop_zerovar(y, X):
    X, _ = drop_zero_variance_columns(X, X)
    return y, X, X


def add_onehot(y, X):
    cat_cols = detect_categorical_cols(X)
    for c in sorted(cat_cols, reverse=True):
        X, _ = one_hot_encode_column(X, X, c)
    return y, X, X


def add_drop_lin_dep_columns(y, X):
    X, _ = drop_linearly_dependent_categorical_columns(X, X)
    return y, X, X


def add_impute_with_median(y, X):
    X, _ = apply_median_imputer(X, X)
    return y, X, X


# ---------- Run experiment ----------
f1_scores = []
labels = []

# Baseline (imputation + standardization + bias)
y_proc, X_proc = baseline_without_pca(y_train, deepcopy(x_train))
baseline_f1 = evaluate_f1(y_proc, X_proc)
f1_scores.append(baseline_f1)
labels.append("Baseline (Impute + Std + Bias)")
print(f"Baseline -> mean F1 = {baseline_f1:.3f}")

# Add droping high nan and zero variance
X_cur = deepcopy(x_train)
y_cur = np.copy(y_train)
y_cur, X_cur, _ = add_drop_highnan(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_drop_zerovar(y_cur, deepcopy(X_cur))
y_proc, X_proc = baseline_without_pca(y_cur, X_cur)
f1_val = evaluate_f1(y_proc, X_proc)
f1_scores.append(f1_val)
labels.append("Drop high nan and zero variance columns")
print(f"step 1 -> {f1_val}")

# Add make special_codes nan
X_cur = deepcopy(x_train)
y_cur = np.copy(y_train)
y_cur, X_cur, _ = add_make_special_codes_nan(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_drop_highnan(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_drop_zerovar(y_cur, deepcopy(X_cur))
y_proc, X_proc = baseline_without_pca(y_cur, X_cur)
f1_val = evaluate_f1(y_proc, X_proc)
f1_scores.append(f1_val)
labels.append("Make special_codes NaN")
print(f"step 2 -> {f1_val}")


# Add one hot encoding and dropping linearly dependent categorical columns
X_cur = deepcopy(x_train)
y_cur = np.copy(y_train)
y_cur, X_cur, _ = add_make_special_codes_nan(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_drop_highnan(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_drop_zerovar(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_onehot(y_cur, deepcopy(X_cur))
y_proc, X_proc = baseline_without_pca(y_cur, X_cur)
f1_val = evaluate_f1(y_proc, X_proc)
f1_scores.append(f1_val)
labels.append("One hot encode categorical columns")
print(f"step 3 -> {f1_val}")

#  dropping linearly dependent categorical columns
X_cur = deepcopy(x_train)
y_cur = np.copy(y_train)
y_cur, X_cur, _ = add_make_special_codes_nan(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_drop_highnan(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_drop_zerovar(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_onehot(y_cur, deepcopy(X_cur))
y_cur, X_cur, _ = add_drop_lin_dep_columns(y_cur, deepcopy(X_cur))
y_proc, X_proc = baseline_without_pca(y_cur, X_cur)
f1_val = evaluate_f1(y_proc, X_proc)
f1_scores.append(f1_val)
labels.append("drop linearly dependent columns")
print(f"step 4 -> {f1_val}")

# impute continuous with median
steps = [
    ("Make special_codes NaN", add_make_special_codes_nan),
    ("Drop high-NaN cols", add_drop_highnan),
    ("Drop zero-var cols", add_drop_zerovar),
    ("One-hot encode categorical", add_onehot),
    ("Drop Linearly dependent cols", add_drop_lin_dep_columns),
    ("Impute continuous columns with median", add_impute_with_median),
]
X_cur = deepcopy(x_train)
y_cur = np.copy(y_train)
for name, fn in steps:
    # apply previous + this step cumulatively on x_train
    y_cur, X_cur, _ = fn(y_cur, deepcopy(X_cur))
# then reapply baseline preprocessing (since it's always included)
y_proc, X_proc = baseline_without_pca(y_cur, X_cur)
f1_val = evaluate_f1(y_proc, X_proc)
f1_scores.append(f1_val)
labels.append("impute continous columns with median")
print(f"step 5 -> {f1_val}")

# add pca
steps = [
    ("Make special_codes NaN", add_make_special_codes_nan),
    ("Drop high-NaN cols", add_drop_highnan),
    ("Drop zero-var cols", add_drop_zerovar),
    ("One-hot encode categorical", add_onehot),
    ("Drop Linearly dependent cols", add_drop_lin_dep_columns),
    ("Impute continuous columns with median", add_impute_with_median),
]
X_cur = deepcopy(x_train)
y_cur = np.copy(y_train)
for name, fn in steps:
    # apply previous + this step cumulatively on x_train
    y_cur, X_cur, _ = fn(y_cur, deepcopy(X_cur))
# then reapply baseline preprocessing (since it's always included)
y_proc, X_proc = baseline_with_pca(y_cur, X_cur)
f1_val = evaluate_f1(y_proc, X_proc)
f1_scores.append(f1_val)
labels.append("Add pca")
print(f"step 6 -> {f1_val}")

# add fill features
steps = [
    ("Fill important features", add_fill_features),
    ("Make special_codes NaN", add_make_special_codes_nan),
    ("Drop high-NaN cols", add_drop_highnan),
    ("Drop zero-var cols", add_drop_zerovar),
    ("One-hot encode categorical", add_onehot),
    ("Drop Linearly dependent cols", add_drop_lin_dep_columns),
    ("Impute continuous columns with median", add_impute_with_median),
]
X_cur = deepcopy(x_train)
y_cur = np.copy(y_train)
for name, fn in steps:
    # apply previous + this step cumulatively on x_train
    y_cur, X_cur, _ = fn(y_cur, deepcopy(X_cur))
# then reapply baseline preprocessing (since it's always included)
y_proc, X_proc = baseline_with_pca(y_cur, X_cur)
f1_val = evaluate_f1(y_proc, X_proc)
f1_scores.append(f1_val)
labels.append("Fill important features")
print(f"Fill important features -> mean F1 = {f1_val:.3f}")

all_steps = [
    ("Fill important features", add_fill_features),
    ("Remove useless columns", add_remove_useless),
    ("Make special_codes NaN", add_make_special_codes_nan),
    ("Drop high-NaN cols", add_drop_highnan),
    ("Drop zero-var cols", add_drop_zerovar),
    ("One-hot encode categorical", add_onehot),
    ("Drop Linearly dependent cols", add_drop_lin_dep_columns),
    ("Impute continuous columns with median", add_impute_with_median),
]
X_cur = deepcopy(x_train)
y_cur = np.copy(y_train)
for name, fn in all_steps:
    # apply previous + this step cumulatively on x_train
    y_cur, X_cur, _ = fn(y_cur, deepcopy(X_cur))

# then reapply baseline preprocessing (since it's always included)
y_proc, X_proc = baseline_with_pca(y_cur, X_cur)
f1_val = evaluate_f1(y_proc, X_proc)
f1_scores.append(f1_val)
labels.append("Remove useless columns [1:9]")
print(f"Remove useless columns -> mean F1 = {f1_val:.3f}")
# Plot results

idx = np.arange(len(f1_scores))

plt.figure(figsize=(11, 6))

# Line plot (absolute F1)
plt.plot(
    idx,
    f1_scores,
    marker="o",
    linewidth=2,
    markersize=7,
    color="teal",
    label="F1-score",
)
plt.fill_between(idx, f1_scores, color="teal", alpha=0.10)

# Annotate exact F1 values above points
for i, v in enumerate(f1_scores):
    plt.annotate(
        f"{v:.3f}",
        (i, v),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        color="teal",
    )

# X axis: diagonal labels for clarity
plt.xticks(idx, labels, rotation=60, ha="right")

# Y axis: zoom to highlight small differences
min_f1, max_f1 = float(np.min(f1_scores)), float(np.max(f1_scores))
rng = max(max_f1 - min_f1, 1e-3)  # avoid zero range
margin = 0.4 * rng
plt.ylim(min_f1 - margin, max_f1 + margin)

plt.ylabel("Mean Validation F1-score")
plt.title("Preprocessing Steps: F1 Trend")
plt.grid(True, axis="y", alpha=0.3)

# Layout & save
plt.tight_layout()
plt.savefig("f1_trend.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Figure saved as 'f1_trend.png'")
