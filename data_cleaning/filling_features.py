import numpy as np
import sys
import os

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("data_cleaning"), "..")))

from data_cleaning.col_helper import *
from helpers import *
from implementations import *

# This class fills columns that would be dropped by drop_high_nan_columns with meaningful values, so we fill them before dropping.


def fill_cardio_rehab_features(x_train, x_test):
    """
    Fill/impute cardio/rehab-related features with sensible defaults.

    Args :
        data : ndarray

    Returns
    -------
    data : same object, modified in place and returned for chaining.
    """
    fill_col(x_train, "CVDASPRN", nan_to_zero=True)
    fill_col(x_test, "CVDASPRN", nan_to_zero=True)
    fill_col(x_train, "HAREHAB1", nan_to_zero=True)
    fill_col(x_test, "HAREHAB1", nan_to_zero=True)
    fill_col(x_train, "STREHAB1", nan_to_zero=True)
    fill_col(x_test, "STREHAB1", nan_to_zero=True)
    fill_col(x_train, "WTCHSALT", nan_to_zero=True)
    fill_col(x_test, "WTCHSALT", nan_to_zero=True)
    fill_col(x_train, "DRADVISE", nan_to_zero=True)
    fill_col(x_test, "DRADVISE", nan_to_zero=True)


def fill_diabetes_features(x_train, x_test):
    """
    Fill/impute diabetes-related features with sensible defaults.

    Args :
        data : ndarray

    Returns
    -------
    data : same object, modified in place and returned for chaining.
    """
    fill_col(x_train, "DIABAGE2", nan_to_median=True)
    fill_col(x_test, "DIABAGE2", nan_to_median=True)
    fill_col(x_train, "DIABEDU", nan_to_zero=True)
    fill_col(x_test, "DIABEDU", nan_to_zero=True)
    fill_col(x_train, "PREDIAB1", nan_to_zero=True)
    fill_col(x_test, "PREDIAB1", nan_to_zero=True)


def fill_all_important_features(x_train, x_test):
    """
    Fill features with sensible defaults.

    Args :
        data : ndarray

    Returns
    -------
    data : same object, modified in place and returned for chaining.
    """
    fill_cardio_rehab_features(x_train, x_test)
    fill_diabetes_features(x_train, x_test)
    return x_train, x_test
