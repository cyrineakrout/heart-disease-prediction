import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from data_cleaning.col_clean import *
from helpers import load_csv_data
from implementations import *
from helpers import *
from implementations import *
from cross_validation import *
from data_preprocessing import *

FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)


def nan_fraction_per_col(X):
    """
    Fraction of NaNs per column, treating inf/-inf as NaN.
    Returns: array shape (D,)
    """
    Xf = X.astype(float, copy=False)
    return np.mean(np.isnan(Xf), axis=0)


def count_uniques_non_nan(a):
    """
    Number of unique non-NaN, finite values in a 1D array.
    Returns : int
    """
    b = a[~np.isnan(a)]
    return np.unique(b).size


def split_categorical_continuous(X, max_unique=7):
    """
    ≤ max_unique -> categorical-like; else continuous-like.
    Returns: cat_mask (bool[D]), cont_mask (bool[D]), uniq_counts (int[D])
    """
    D = X.shape[1]
    uniq = np.zeros(D, dtype=int)
    for j in range(D):
        uniq[j] = count_uniques_non_nan(X[:, j])
    cat_mask = uniq <= max_unique
    cont_mask = uniq > max_unique
    return cat_mask, cont_mask, uniq


def corr_with_y_per_feature(X, y):
    """
    Pearson corr(feature_j, y) per column using only rows where both are NOT NaN.
    Returns array length D with NaN where not computable (zero variance or <2 points).
    """
    N, D = X.shape
    out = np.full(D, np.nan, dtype=float)
    for j in range(D):
        x = X[:, j].astype(float, copy=False)
        mask = ~np.isnan(x) & ~np.isnan(y)
        n = int(mask.sum())
        if n < 2:
            continue
        xj = x[mask]
        yj = y[mask]
        sx = np.std(xj)
        sy = np.std(yj)
        if sx == 0 or sy == 0:
            continue
        cov = np.mean((xj - xj.mean()) * (yj - yj.mean()))
        out[j] = cov / (sx * sy)
    return out


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Load data
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        "dataset", sub_sample=False
    )

    X = x_train.astype(float, copy=False)
    y = y_train.astype(float, copy=False)

    N, D = X.shape

    # ---------- 1) Class imbalance (-1 vs +1) ----------
    cnt_neg = int(np.sum(y == -1))
    cnt_pos = int(np.sum(y == 1))
    plt.figure()
    plt.bar([-1, 1], [cnt_neg, cnt_pos])
    plt.title("Class balance (-1 vs +1)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks([-1, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_balance_pm1.png"), dpi=200)
    plt.close()

    # ---------- 2) NaN fraction per column (top 30) ----------
    nan_frac = nan_fraction_per_col(X)  # shape (D,)
    edges = np.array([0.0, 0.25, 0.50, 0.75, 1.0000001])
    labels = ["0–25%", "25–50%", "50–75%", "75–100%"]

    counts, _ = np.histogram(nan_frac, bins=edges)
    tot = counts.sum()

    plt.figure()
    plt.bar(labels, counts)
    for i, c in enumerate(counts):
        pct = 100.0 * c / tot if tot else 0.0
        plt.text(i, c, f"{c}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.ylabel("# of columns")
    plt.title("Columns by NaN fraction bucket")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "nan_fraction_buckets.png"), dpi=200)
    plt.close()

    # ---------- 3) Categorical vs Continuous (unique-count rule) ----------
    cat_mask, cont_mask, uniq_counts = split_categorical_continuous(X, max_unique=12)
    n_cat = int(np.sum(cat_mask))
    n_cont = int(np.sum(cont_mask))

    # distribution of unique counts
    low_cap = 30
    vals = uniq_counts.copy()
    vals = vals[~np.isnan(vals)]
    vals = vals.astype(int)

    # counts for 1..low_cap, plus overflow
    bins = np.arange(1, low_cap + 1)
    counts_low = np.array([(vals == k).sum() for k in bins])
    overflow = int((vals > low_cap).sum())

    plt.figure()
    plt.bar(bins, counts_low, align="center")
    plt.bar(low_cap + 1, overflow, align="center")  # '30+' bar
    plt.xticks(
        list(bins[::3]) + [low_cap + 1],
        [str(k) for k in bins[::3]] + [f"{low_cap}+"],
        rotation=0,
    )
    for x, c in zip(list(bins) + [low_cap + 1], list(counts_low) + [overflow]):
        if c > 0:
            plt.text(x, c, str(c), ha="center", va="bottom", fontsize=8)

    plt.xlabel("Unique non-NaN values per feature")
    plt.ylabel("#features")
    plt.title(f"Unique counts (fine view up to {low_cap}, plus '{low_cap}+')")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "unique_counts_lowend.png"), dpi=200)
    plt.close()

    # ---------- 4) |corr(feature, y)| distribution ----------
    continuous_columns = detect_continuous_cols(X)
    print(continuous_columns.size)
    corrs = corr_with_y_per_feature(X[:, continuous_columns], y)  # y in {-1,+1}
    abs_corrs = np.abs(corrs[~np.isnan(corrs)])
    bins = 40 if abs_corrs.size >= 200 else max(10, abs_corrs.size // 5 + 1)

    plt.figure()
    plt.hist(abs_corrs, bins=bins)
    plt.xlabel("|corr(feature, y)|")
    plt.ylabel("#features")
    plt.title("Distribution of absolute feature–label correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "abs_feature_corr_hist.png"), dpi=200)
    plt.close()
