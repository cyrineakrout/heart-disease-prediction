import numpy as np


def choose_d_by_variance(X, threshold=0.95):
    """
    Automatically choose the number of PCA components
    to reach the desired explained variance threshold.

    Args:
        X (np.ndarray): Training data, shape (N, D)
        threshold (float): Desired explained variance ratio (0–1)

    Returns:
        d (int): number of components
        cum_explained (np.ndarray): cumulative explained variance
    """
    # Center data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Covariance matrix
    C = np.cov(X_centered, rowvar=False)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]  # sort descending
    eigvecs = eigvecs[:, order]

    # Compute cumulative explained variance
    cum_explained = np.cumsum(eigvals) / np.sum(eigvals)

    # Choose smallest d that reaches threshold
    d = np.searchsorted(cum_explained, threshold) + 1

    return d, cum_explained


class PCA(object):
    """
    PCA dimensionality reduction class.

    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d

        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT:
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        N = training_data.shape[0]
        # Compute the mean of data
        self.mean = np.mean(training_data, axis=0)
        # Center the data with the mean
        training_data_centered = training_data - self.mean
        # Create the covariance matrix
        C = np.cov(training_data_centered, rowvar=False)
        # Compute the eigenvectors and eigenvalues. Hint: look into np.linalg.eigh()
        eigvals, eigvecs = np.linalg.eigh(C)
        # Choose the top d eigenvalues and corresponding eigenvectors.
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        self.W = eigvecs[:, : self.d]
        eg = eigvals[: self.d]

        # Compute the explained variance
        exvar = np.sum(eg) / np.sum(eigvals) * 100

        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        data_reduced = (data - self.mean) @ self.W

        return data_reduced
