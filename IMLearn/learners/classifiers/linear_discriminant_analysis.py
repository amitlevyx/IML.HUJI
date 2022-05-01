from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.fitted_ = True
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        K = self.classes_.shape[0]
        self.mu_ = np.zeros((K, n_features))
        self.pi_ = np.zeros(K)
        for k in range(K):
            X_k = X[y == self.classes_[k]]
            self.mu_[k] = np.mean(X_k, axis=0)
            self.pi_[k] = np.mean(y == self.classes_[k])
        summ = np.zeros((n_features, n_features))
        for i in range(n_samples):
            inner_factor = np.array([X[i] - self.mu_[y[i]]])
            summ += inner_factor.T @ inner_factor
        self.cov_ = summ / (n_samples - K)
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        n_samples = X.shape[0]
        likelihood = self.likelihood(X)
        y_hat = np.zeros(n_samples, dtype=int)
        for sample in range(n_samples):
            pos_y = self.classes_[np.argmax(likelihood[sample])]
            y_hat[sample] = pos_y
        return y_hat

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        K = self.classes_.shape[0]
        m = X.shape[0]
        d = X.shape[1]
        likelihood = np.zeros((m, K))
        for sample in range(m):
            sample_likelihood = []
            for k in range(K):
                mu_k = self.mu_[k]
                inner_factor = X[sample] - mu_k
                cur_likelihood = np.log(self.pi_[k]) - (d / 2) * np.log(2 * np.pi) - 0.5 * np.log(det(self.cov_))
                cur_likelihood += -0.5 * (inner_factor.T @ self._cov_inv @ inner_factor)
                sample_likelihood.append(cur_likelihood)
            likelihood[sample] = sample_likelihood
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return loss_functions.misclassification_error(y, self.predict(X))
