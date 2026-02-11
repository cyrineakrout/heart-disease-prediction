import numpy as np
from helpers import *
from implementations import *
from data_cleaning.pca import *
from cross_validation import *
from data_cleaning.filling_features import *
from data_cleaning.col_clean import *


def process_data(y_train, x_train, x_test):
    # fill important features that would be dropped otherwise because of high nan fraction
    x_train, x_test = fill_all_important_features(x_train, x_test)

    # remove useless features
    x_train = np.c_[x_train[:, 0], x_train[:, 9:]]
    x_test = np.c_[x_test[:, 0], x_test[:, 9:]]

    # convert -1 to 0
    y_train[y_train == -1] = 0

    # replace special-coded values (like 9999) with NaN.
    x_train, x_test = make_special_codes_nan(x_train, x_test)

    # removing columns with high rate of nan values.
    x_train, x_test = drop_high_nan_columns(x_train, x_test)

    # remove columns that have zero-variance (no variability = useless feature).
    x_train, x_test = drop_zero_variance_columns(x_train, x_test)

    # identify which columns are categorical vs continuous.
    categorical_columns = detect_categorical_cols(x_train)
    continuous_columns = detect_continuous_cols(x_train)

    # expand categorical columns into one-hot binary features.
    for col in sorted(categorical_columns, reverse=True):
        x_train, x_test = one_hot_encode_column(x_train, x_test, col)

    # drop_linearly_dependent_categorical_columns after one-hot encoding.
    x_train, x_test = drop_linearly_dependent_categorical_columns(x_train, x_test)
    categorical_columns = detect_categorical_cols(x_train)
    continuous_columns = detect_continuous_cols(x_train)

    # fills remaining NaNs with median of each column.
    x_train, x_test = apply_median_imputer(x_train, x_test)

    # standardize dataset
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # apply PCA
    d_opt, cumvar = choose_d_by_variance(x_train[:, continuous_columns], threshold=0.95)
    pca_obj = PCA(d_opt)
    pca_obj.find_principal_components(x_train[:, continuous_columns])
    x_train = np.c_[
        x_train[:, categorical_columns],
        pca_obj.reduce_dimension(x_train[:, continuous_columns]),
    ]
    x_test = np.c_[
        x_test[:, categorical_columns],
        pca_obj.reduce_dimension(x_test[:, continuous_columns]),
    ]
    # add bias
    x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    return y_train, x_train, x_test
