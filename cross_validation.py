import numpy as np
from implementations import *
from functions import *


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true : array-like of shape (N,), ground-truth labels
        y_pred : array-like of shape (N,), predicted labels

    Returns:
        float in [0,1] = (# correct predictions) / N
    """
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred):
    """
    Compute F1 score manually.

    Args:
        y_true : array-like of shape (N,), ground-truth labels
        y_pred : array-like of shape (N,), predicted labels

    Returns:
        f1_score = 2 * precision * recall / (precision + recall)
    """
    # convert to {0,1} if needed
    y_true_bin = (y_true == 1).astype(int)
    y_pred_bin = (y_pred == 1).astype(int)

    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return f1


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_one_fold(
    y,
    x,
    k_indices,
    k,
    lambda_,
    gamma,
    max_iters=1000,
    initial_w=None,
    threshold_grid=None,
):
    """
    Train regularized logistic regression on K-1 folds and evaluate on the held-out fold.

    Args:
        y: (N,) labels (expected in {0,1})
        x: (N, D) features
        k_indices: 2D array from build_k_indices(...)
        k: int, index of the test fold
        lambda_: float, L2 regularization strength (0.0 -> plain logistic)
        gamma: float, learning rate
        max_iters: int
        initial_w: np.ndarray or None
        threshold_grid: iterable of thresholds to sweep; if None, use fixed 0.22

    Returns:
        If threshold_grid is not None:
            (best_f1_te, best_t_te)
        Else (fixed threshold 0.22):
            (f1_te, acc_te)
    """
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # if lambda = 0 it will train logistic regression
    w, loss = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)

    # for logistic regression with threshold tuning
    if threshold_grid is not None:
        p_te = sigmoid(x_te.dot(w))

        # threshold sweep to maximize F1 on each split
        best_f1_te, best_t_te = -1.0, 0.5
        for t in threshold_grid:
            yhat_te = (p_te >= t).astype(int)
            f1_te = f1_score(y_te, yhat_te)

            if f1_te > best_f1_te:
                best_f1_te, best_t_te = f1_te, t

        return best_f1_te, best_t_te
    # for regularized logistic regression without threshold tuning
    else:
        y_hat = sigmoid(x_te @ w)
        f1_score_te = f1_score(y_te, np.where((y_hat >= 0.2), 1, 0))
        accuracy_te = accuracy(y_te, np.where((y_hat >= 0.2), 1, 0))
        return f1_score_te, accuracy_te


def cv_logistic_tune_gamma_and_threshold(
    y, x, k_fold, gammas, threshold_grid, seed=12, max_iters=1000
):
    """
    Stage A. Logistic regression without L2 penalty (lambda=0):
    Grid over gamma and threshold. Pick (gamma*, threshold*) maximizing mean CV F1.
    """
    if threshold_grid is None:
        threshold_grid = np.linspace(0, 1, 101)  # default threshold sweep grid
    k_indices = build_k_indices(y, k_fold, seed)
    best = {"gamma": None, "threshold": None, "cv_f1": -1.0}

    for gamma in gammas:
        f1s, ts = [], []
        for k in range(k_fold):

            f1, t = cross_validation_one_fold(
                y,
                x,
                k_indices,
                k,
                lambda_=0.0,
                gamma=gamma,
                threshold_grid=threshold_grid,
                max_iters=max_iters,
                initial_w=np.zeros(x.shape[1]),
            )
            f1s.append(f1)
            ts.append(t)
        mean_f1 = float(np.mean(f1s))
        mean_t = float(np.mean(ts))
        if mean_f1 > best["cv_f1"]:
            best.update({"gamma": gamma, "threshold": mean_t, "cv_f1": mean_f1})
    return best  # {"gamma": ..., "threshold": ..., "cv_f1": ...}


def cv_reg_logistic_tune_gamma_lambda_threshold(
    y, x, k_fold, lambdas, gammas, seed=12, max_iters=1000, threshold_grid=None
):
    """
    Tune gamma and lambda together by CV, maximizing F1.

    For each regularization strength `lambda_` and learning rate `gamma`,
    this function:
        builds K CV folds (currently non-stratified),
        trains on K-1 folds, validates on the held-out fold,
        collects validation F1 and accuracy per fold,
        aggregates mean validation F1 per gamma,
        picks the best gamma for that lambda,
        finally selects the (lambda, gamma) pair with the highest mean val F1.


    Args:
        y : array-like, shape (N,)
        x : array-like, shape (N, D)
        k_fold : int
        lambdas : sequence of float
        gammas : sequence of float
        seed : int, default=12
        max_iters : int, default=1000


    Returns:
        best_lambda, best_gamma, stats dict
    """
    if threshold_grid is None:
        threshold_grid = np.linspace(0, 1, 101)
    k_indices = build_k_indices(y, k_fold, seed)

    best_f1_te_per_lambda = []
    best_gammas = []
    best_thresholds = []

    for lambda_ in lambdas:
        f1_te, thresholds = [], []
        for gamma in gammas:
            f1_te_tmp, thresholds_tmp = [], []
            for k in range(k_fold):
                f1_val, threshold = cross_validation_one_fold(
                    y,
                    x,
                    k_indices,
                    k,
                    lambda_,
                    gamma,
                    initial_w=np.zeros(x.shape[1]),
                    max_iters=max_iters,
                    threshold_grid=threshold_grid,
                )
                f1_te_tmp.append(f1_val)
                thresholds_tmp.append(threshold)
            print("f1score on val is ", np.mean(f1_te_tmp))
            f1_te.append(np.mean(f1_te_tmp))
            thresholds.append(np.mean(thresholds_tmp))
        ind_gama_opt = np.argmax(f1_te)
        best_gammas.append(gammas[ind_gama_opt])
        best_f1_te_per_lambda.append(f1_te[ind_gama_opt])
        best_thresholds.append(thresholds[ind_gama_opt])

    # pick gamma with best mean test F1
    best_idx = int(np.argmax(best_f1_te_per_lambda))
    best_lambda = lambdas[best_idx]
    best_gamma = best_gammas[best_idx]
    best_threshold = best_thresholds[best_idx]
    best_cv_f1 = float(best_f1_te_per_lambda[best_idx])

    stats = {
        "best_lambda": best_lambda,
        "best_gamma": best_gamma,
        "best_threshold": best_threshold,
        "best_cv_f1": best_cv_f1,
    }

    return best_lambda, best_gamma, stats
