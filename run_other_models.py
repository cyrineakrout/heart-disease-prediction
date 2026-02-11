import numpy as np
from helpers import *
from implementations import *
from cross_validation import *
from data_preprocessing import *
import argparse

# --- Command-line argument parsing ---------------------------------
# Allows running: python run.py --mode full --model reglogreg
parser = argparse.ArgumentParser(description="Run training for project1 models 1-5")
parser.add_argument(
    "--model",
    choices=[
        "mean_square_gd",
        "mean_square_sgd",
        "least_squares",
        "ridge_regression",
        "logistic_regression",
    ],
    default="mean_square_gd",
    help="Model to train",
)

args = parser.parse_args()

MODEL = args.model
MAX_ITERS = 2000


# loading data
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
    "dataset", sub_sample=False
)

# preprocessing data
y_train, x_train, x_test = process_data(y_train, x_train, x_test)

if MODEL == "mean_square_gd":
    # Convert 0 back to -1, since -1 was transformed to 0 during preprocessing

    y_train[y_train == 0] = -1
    # train on full data
    weights, loss = mean_squared_error_gd(
        y_train,
        x_train,
        initial_w=np.zeros(x_train.shape[1]),
        max_iters=MAX_ITERS,
        gamma=0.0001,
    )
    y_pred = np.sign(x_test @ weights)
    # generate predictions
    create_csv_submission(test_ids, y_pred, "submission.csv")

elif MODEL == "mean_square_sgd":
    # Convert 0 back to -1, since -1 was transformed to 0 during preprocessing
    y_train[y_train == 0] = -1
    # train on full data
    weights, loss = mean_squared_error_sgd(
        y_train,
        x_train,
        initial_w=np.zeros(x_train.shape[1]),
        max_iters=MAX_ITERS,
        gamma=0.0001,
    )
    y_pred = np.sign(x_test @ weights)
    # generate predictions
    create_csv_submission(test_ids, y_pred, "submission.csv")

elif MODEL == "least_squares":
    # Convert 0 back to -1, since -1 was transformed to 0 during preprocessing
    y_train[y_train == 0] = -1
    # train on full data
    weights, loss = least_squares(y_train, x_train)
    y_pred = np.sign(x_test @ weights)
    # generate predictions
    create_csv_submission(test_ids, y_pred, "submission.csv")

elif MODEL == "ridge_regression":
    # Convert 0 back to -1, since -1 was transformed to 0 during preprocessing
    y_train[y_train == 0] = -1
    # train on full data
    weights, loss = ridge_regression(y_train, x_train, lambda_=0.1)
    y_pred = np.sign(x_test @ weights)
    # generate predictions
    create_csv_submission(test_ids, y_pred, "submission.csv")

elif MODEL == "logistic_regression":
    weights, loss = logistic_regression(
        y_train,
        x_train,
        initial_w=np.zeros(x_train.shape[1]),
        max_iters=MAX_ITERS,
        gamma=0.1,
    )
    probs = sigmoid(x_test @ weights)
    y_pred = np.where((probs >= 0.212), 1, -1)
    # generate predictions
    create_csv_submission(test_ids, y_pred, "submission.csv")
