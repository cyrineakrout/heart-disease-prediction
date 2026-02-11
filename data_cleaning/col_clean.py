import numpy as np
import sys
import os

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("data_cleaning"), "..")))
from helpers import *
from implementations import *
from data_cleaning.pca import *
from cross_validation import *
from data_cleaning.filling_features import *


def one_hot_encode_column(Xtr, Xte, col):
    """
    One-hot encode a single column if 3 <= #classes <= max_classes (skip binaries and huge-cardinality) then drop original column.

    Args
    ----
    Xtr : (Ntr, D) array
    Xte : (Nte, D) array
    col : int
        Index of the column to encode (in the current Xtr/Xte layout).
    Returns
    -------
    Xtr, Xte : arrays with the original `col` removed and one-hots appended.

    """

    classes = np.unique(Xtr[:, col])
    if classes.size > 2 and classes.size <= 7:
        one_hot_labels_tr = np.zeros([Xtr.shape[0], classes.size])
        one_hot_labels_te = np.zeros([Xte.shape[0], classes.size])
        for i in range(classes.size):
            if np.isnan(classes[i]):
                one_hot_labels_tr[np.where(np.isnan(Xtr[:, col])), i] = 1
                one_hot_labels_te[np.where(np.isnan(Xte[:, col])), i] = 1
            else:
                one_hot_labels_tr[np.where(Xtr[:, col] == classes[i]), i] = 1
                one_hot_labels_te[np.where(Xte[:, col] == classes[i]), i] = 1
        Xtr = np.delete(Xtr, col, axis=1)
        Xte = np.delete(Xte, col, axis=1)
        Xtr = np.c_[Xtr, one_hot_labels_tr]
        Xte = np.c_[Xte, one_hot_labels_te]
    return Xtr, Xte


def make_special_codes_nan(Xtr, Xte):
    """
    Replace known 'code' values (77, 99, 999, ...) by NaN in both train and test.

    For each column:
      - Use train column to decide which codes are 'above the real max' (via max_and_special_codes),
      - Mark those code values as NaN in both train and test.

    Returns
    Xtr, Xte : copies of inputs with special codes replaced by NaN (float dtype)
    """

    Xtr_copy = Xtr.astype(float, copy=True)
    Xte_copy = Xte.astype(float, copy=True)
    for col in range(Xtr.shape[1]):
        maxi, special_code = max_and_special_codes(Xtr_copy[:, col])
        mask_tr = np.isin(Xtr_copy[:, col], special_code)
        mask_te = np.isin(Xte_copy[:, col], special_code)
        Xtr[mask_tr, col] = np.nan
        Xte[mask_te, col] = np.nan
    return Xtr, Xte


def drop_high_nan_columns(Xtr, Xte, threshold=0.6):
    """
    - Maps coded missings -> NaN (if provided).
    - Computes NaN fraction per column on TRAIN only.
    - Drops columns with NaN frac > threshold from both train & test.
    """
    Xtr = Xtr.astype(float, copy=True)
    Xte = Xte.astype(float, copy=True)

    nan_frac = np.mean(np.isnan(Xtr), axis=0)  # train-only
    keep_mask = nan_frac <= threshold
    Xtr_new = Xtr[:, keep_mask]
    Xte_new = Xte[:, keep_mask]
    # print(f"Dropped {np.size(keep_mask) - int(keep_mask.sum())} of nan columns")

    return Xtr_new, Xte_new


def drop_zero_variance_columns(Xtr, Xte, tol=1e-12):
    """
    - Computes std on TRAIN only, ignoring NaNs.
    - Drops columns with std <= tol or std is NaN (all-NaN / constant).
    - Applies same mask to TEST if provided.
    """
    Xtr = Xtr.astype(float, copy=True)
    std = np.nanstd(Xtr, axis=0)
    keep_mask = np.isfinite(std) & (std > tol)

    Xtr_out = Xtr[:, keep_mask]
    Xte_out = Xte[:, keep_mask]

    # print(
    #     f"Dropped {np.size(keep_mask) - int(keep_mask.sum())} zero-variance cols "
    #     f"of {len(keep_mask)} total."
    # )

    return (Xtr_out, Xte_out)


def drop_linearly_dependent_categorical_columns(x_train, x_test):
    """Drop linearly dependent columns from the training set and apply the same mask to the test set."""

    categorical_columns = detect_categorical_cols(x_train)
    continuous_columns = detect_continuous_cols(x_train)
    X_cat_tr = x_train[:, categorical_columns]
    Q, R = np.linalg.qr(X_cat_tr)
    keep_mask = np.abs(np.diag(R)) > 1e-10
    keep_cat_idx_local = np.where(keep_mask)[0]
    keep_cat_idx_global = np.array(categorical_columns)[keep_cat_idx_local]

    x_train = np.c_[x_train[:, keep_cat_idx_global], x_train[:, continuous_columns]]
    x_test = np.c_[x_test[:, keep_cat_idx_global], x_test[:, continuous_columns]]
    return x_train, x_test


def detect_continuous_cols(X, min_unique=7):
    """
    Returns indices of columns that have more than `min_unique` unique non-NaN values.
    """
    cols = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if np.unique(col).size > min_unique:
            cols.append(j)
    return np.array(cols, dtype=int)


def detect_categorical_cols(X, max_unique=7):
    """
    Returns indices of columns that have more than `min_unique` unique non-NaN values.
    """
    cols = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if np.unique(col).size <= max_unique:
            cols.append(j)
    return np.array(cols, dtype=int)


# apply only on continuous columns since categorical columns have been one-hot encoded
def apply_median_imputer(x_train, x_test):
    """
    Median-impute ONLY continuous columns in X (in place), leaving categorical columns unchanged.
    Missing values in continuous features are imputed using the median of the training set. The same medians are applied to the test set.

    Args:

    x_train : (N1, D) np.ndarray
    x_test : (N2, D) np.darray

    Returns
    -------
    x_train : (N1, D) np.ndarray, the same array object, after median imputation on continuous columns.
    x_test : (N2, D) np.ndarray, the same array object, after median imputation on continuous columns.
    """

    continuous_columns = detect_continuous_cols(x_train)
    medians = np.nanmedian(x_train[:, continuous_columns], axis=0)

    for idx, col in enumerate(continuous_columns):
        # Apply to training set
        mask_train = np.isnan(x_train[:, col])
        x_train[mask_train, col] = medians[idx]

        # Apply to test set
        mask_test = np.isnan(x_test[:, col])
        x_test[mask_test, col] = medians[idx]

    return x_train, x_test
