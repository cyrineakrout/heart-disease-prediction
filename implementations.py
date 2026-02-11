import numpy as np
from functions import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vecotr of the SGD method of shape (D,).
        loss: the last loss of the SGD method corresponding to the weight vector w, a scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        # compute a gradient
        gradient = compute_gradient(y, tx, w)
        # update w through the stochastic gradient update
        w = w - gamma * gradient
    # calculate the loss
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vecotr of the SGD method of shape (D,).
        loss: the last loss of the SGD method corresponding to the weight vector w, a scalar.

    """
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, x_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient
            gradient = compute_gradient(y_batch, x_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * gradient
    # calculate the loss
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """
    Calculate the least squares solution to y = tx * w.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
    Returns:
        w: optimal weights, numpy array of shape (D,)
        mse: mean squared error of the solution
    """
    # Normal equation: w = solve(tx.T @ tx, tx.T @ y) for better numerical stability
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - tx @ w
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar for regularization parameter.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    lambda_p = 2 * len(y) * lambda_
    a = tx.T.dot(tx) + lambda_p * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of the method
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vecotr of the method of shape (D,).
        loss: the last loss of the method corresponding to the weight vector w, a scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        # compute the gradient
        gradient = compute_logistic_regression_gradient(y, tx, w)
        # update w through the gradient update
        w = w - gamma * gradient
    # calculate the loss
    loss = compute_logistic_regression_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        lambda_: scalar for regularization parameter
        initial_w: initial weights, numpy array of shape (D,)
        max_iters: number of iterations for gradient descent
        gamma: learning rate
    Returns:
        w: optimal weights, numpy array of shape (D,)
        loss: final loss value
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_reg_logistic_regression_gradient(y, tx, w, lambda_)
        w -= gamma * gradient
    loss = compute_logistic_regression_loss(y, tx, w)
    return w, loss
