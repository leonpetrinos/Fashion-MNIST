import numpy as np

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

        # mean of the data
        self.mean = np.mean(training_data, axis=0)
        # centered data with the mean
        X_tilde = training_data - self.mean
        # covariance matrix
        C = X_tilde.T @ X_tilde 
        # eigenvalues and eigenvectors of covariance matrix
        eigvals, eigvecs = np.linalg.eigh(C)
        # find indexes of eigenvalues sorted in decreasing order 
        indexes = np.argsort(-eigvals)
        eigvals = eigvals[indexes] # sorted eigenvals
        eigvecs = eigvecs[:, indexes] # sorted eigenvectors

        self.W = eigvecs[:, :self.d]
        biggest_eigvals = eigvals[:self.d]

        # explained variance
        exvar = 100 * np.sum(biggest_eigvals) / np.sum(eigvals)
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
 
        centered = data - self.mean
        return centered @ self.W
        

