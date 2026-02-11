from dataclasses import dataclass
from typing import List, Optional
import csv
import numpy as np


@dataclass
class ColumnInfo:
    """Simple struct to hold metadata for each feature column."""

    name: str
    excel_col: str
    index: int


# Load only the header row to build column mappings.
with open("dataset/x_train.csv", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)  # read the first line only

FREQ_VALS = np.array(
    [i + 100 for i in range(10)]
    + [i + 200 for i in range(6)]
    + [i + 300 for i in range(30)]
    + [i + 400 for i in range(12)]
)


def index_to_col(index: int) -> str:
    """Convert zero-based integer index to Excel column (A, B, ..., Z, AA, AB...)."""
    col = ""
    index += 1
    while index >= 0:
        index, rem = divmod(index, 26)
        col = chr(65 + rem) + col
        index -= 1
    return col


# Build a list of ColumnInfo for *feature* columns only (skip ID at header[0]).
COLUMNS = [ColumnInfo(name, index_to_col(i), i) for i, name in enumerate(header[1:])]


def index_to_excel_col(index: int) -> Optional[str]:
    """Map feature index -> Excel label. Returns None if index is invalid."""
    for col in COLUMNS:
        if col.index == index:
            return col.excel_col
    return None


def excel_col_to_index(excel_col: str) -> Optional[int]:
    """Map Excel label -> feature index. Returns None if unknown."""
    for col in COLUMNS:
        if col.excel_col == excel_col:
            return col.index
    return None


def excel_col_to_name(excel_col: str) -> Optional[str]:
    """Map Excel label -> CSV column name. Returns None if unknown."""
    for col in COLUMNS:
        if col.excel_col == excel_col:
            return col.name
    return None


def index_to_name(index: int) -> Optional[str]:
    """Map feature index -> CSV column name. Returns None if invalid index."""
    for col in COLUMNS:
        if col.index == index:
            return col.name
    return None


def name_to_index(name: str) -> Optional[int]:
    """Map CSV column name -> feature index. Returns None if name not found."""
    for col in COLUMNS:
        if col.name == name:
            return col.index
    return None


def name_to_excel_col(name: str) -> Optional[str]:
    """Map CSV column name -> Excel label. Returns None if name not found."""
    for col in COLUMNS:
        if col.name == name:
            return col.excel_col
    return None


def print_from_cols(x, cols=None, nb_rows=20, first_row=0):
    """print a small slice of selected columns from a 2D NumPy array `x`."""
    cols = cols or [i for i in range(x.shape[1])]
    arr = np.empty((len(cols), nb_rows + 1), dtype=object)
    for i, col in enumerate(cols):
        vals = x[first_row : nb_rows + first_row, col]
        arr[i, 0] = index_to_name(col)
        arr[i, 1:] = vals
    print(arr.transpose())


def max_and_special_codes(col):
    """
    Replace known placeholder 'code' values with NaN, then compute:
      - maxi: max of the cleaned column (ignoring NaNs)
      - bad_codes_above_max: list of code values that are greater than the observed max

    Returns:
        maxi, [code values > maxi]
    """
    special_codes = [
        7.0,
        8.0,
        9.0,
        77.0,
        88.0,
        99.0,
        555.0,
        777.0,
        888.0,
        999.0,
        5555.0,
        7777.0,
        8888.0,
        9999.0,
        9990.0,
        9998.0,
        99900.0,
        999000.0,
    ]
    col = np.copy(col)
    mask = np.isin(col, special_codes)
    col[mask] = np.nan
    maxi = np.nanmax(col)
    return maxi, [o for o in special_codes if o > maxi]


def fill_col(
    data,
    name,
    nan_to_zero=False,
    nan_to_median=False,
):
    """
    Impute a single column in-place by name.

    Args:
        data        : (N, D) NumPy array of features .
        name        : CSV header of the target column .
        nan_to_zero : if True, replace NaNs with 0.
        nan_to_median: if True, replace NaNs with the column median (ignoring NaNs).

    Returns:
        The imputed NumPy view `data[:, col]` .
    """
    col = name_to_index(name)
    if nan_to_zero:
        data[:, col] = np.where(np.isnan(data[:, col]), 0, data[:, col])
    elif nan_to_median:
        med_val = np.nanmedian(data[:, col])
        data[:, col] = np.where(np.isnan(data[:, col]), med_val, data[:, col])
    return data[:, col]
