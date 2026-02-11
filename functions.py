import numpy as np


def compute_mse_loss(y, tx, w):
    """Calculate the mean squared error.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the MSE loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        A numpy array of shape (D,) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    N = len(y)
    error = y - tx.dot(w)
    gradient = -1 / N * tx.T.dot(error)
    return gradient


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array"""
    return 1 / (1 + np.exp(-t))


def compute_logistic_regression_gradient(y, tx, w):
    """Compute the gradient of loss.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        A numpy array of shape (D,) (same shape as w), containing the gradient of the loss.
    """
    N = len(y)
    pred = sigmoid(tx.dot(w))
    gradient = 1 / N * tx.T.dot(pred - y)
    return gradient


def compute_logistic_regression_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a non-negative loss
    """
    N = len(y)
    pred = sigmoid(tx.dot(w))
    eps = 1e-15  # small constant
    pred = np.clip(pred, eps, 1 - eps)
    loss = -1 / N * (y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)))
    return loss


def compute_reg_logistic_regression_gradient(y, tx, w, lambda_):
    """compute the gradient of the L2 regularized loss.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)
        lambda_: scalar forregularization parameter

    Returns:
        a non-negative loss
    """
    return compute_logistic_regression_gradient(y, tx, w) + 2 * lambda_ * w
