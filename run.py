import numpy as np
from helpers import *
from implementations import *
from cross_validation import *
from data_preprocessing import *
import argparse

# --- Command-line argument parsing ---------------------------------
# Allows running: python run.py --mode full --model reglogreg
parser = argparse.ArgumentParser(
    description="Run training/CV/validation for project1 models"
)
parser.add_argument(
    "--mode",
    choices=["full", "cv", "val"],
    default="full",
    help="Which mode to run: 'full' retrain on all data, 'cv' for cross-validation tuning, 'val' for quick validation split",
)
parser.add_argument(
    "--model",
    choices=["logreg", "reglogreg"],
    default="reglogreg",
    help="Model type to use during CV: 'logreg' or 'reglogreg' (regularized logistic)",
)
args = parser.parse_args()
MODE = args.mode
MODEL = args.model
SEED = 12
K_FOLD = 5
MAX_ITERS = 2000

# grids (used by CV and/or quick val run)
LAMBDAS = np.logspace(-4, -1, 4)
GAMMAS = np.logspace(-4, -1, 4)
THRESH_GRID = np.linspace(0.02, 0.98, 49)
FIXED_THRESHOLD = 0.208

# loading data
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    "dataset", sub_sample=False
)

# preprocessing data
y_train, x_train, x_test = process_data(y_train, x_train, x_test)

if MODE == "full":
    # train regularized logistic regression (chosen model) on full data
    weights, loss = reg_logistic_regression(
        y_train,
        x_train,
        lambda_=0.001,
        initial_w=np.zeros(x_train.shape[1]),
        max_iters=MAX_ITERS,
        gamma=0.1,
    )
    probs = sigmoid(x_test @ weights)
    y_pred = np.where((probs >= FIXED_THRESHOLD), 1, -1)
    # generate predictions
    create_csv_submission(test_ids, y_pred, "submission.csv")

elif MODE == "cv":
    if MODEL == "logreg":
        # ---- Stage A: tune gamma & threshold with lambda=0 (plain logistic)
        stageA = cv_logistic_tune_gamma_and_threshold(
            y_train,
            x_train,
            k_fold=K_FOLD,
            gammas=GAMMAS,
            threshold_grid=THRESH_GRID,
            seed=SEED,
            max_iters=MAX_ITERS,
        )
        tau_star = stageA["threshold"]
        gamma_star_log = stageA["gamma"]
        print(
            f"[CV][Stage A] tau*={tau_star:.3f}, gamma*={gamma_star_log}, cv_f1={stageA['cv_f1']:.4f}"
        )
    if MODEL == "reglogreg":
        # ---- Stage B: tune lambda & gamma & threshold (regularized logistic)
        best_lambda, best_gamma, stats = cv_reg_logistic_tune_gamma_lambda_threshold(
            y_train,
            x_train,
            k_fold=K_FOLD,
            lambdas=LAMBDAS,
            gammas=GAMMAS,
            seed=SEED,
            max_iters=MAX_ITERS,
            threshold_grid=THRESH_GRID,
        )
        print("[CV][Stage B] stats:", stats)


elif MODE == "val":
    seed = np.random.seed(SEED)
    idx = np.random.RandomState(0).permutation(len(y_train))
    cut = int(0.8 * len(y_train))
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr = x_train[tr], y_train[tr]
    Xva, yva = x_train[va], y_train[va]

    # apply reg_logistic_regression to find the weights
    weights, loss = reg_logistic_regression(
        ytr,
        Xtr,
        lambda_=0.001,
        initial_w=np.zeros(x_train.shape[1]),
        max_iters=MAX_ITERS,
        gamma=0.1,
    )

    print(f1_score(yva, np.where(sigmoid(Xva @ weights) >= FIXED_THRESHOLD, 1, 0)))
    print(accuracy(yva, np.where((sigmoid(Xva @ weights) >= FIXED_THRESHOLD), 1, 0)))
