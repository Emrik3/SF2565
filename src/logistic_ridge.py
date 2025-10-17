from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.optimize import minimize

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.exceptions import NotFittedError


class LogRidge(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C

    @staticmethod
    def _sigmoid(z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    @staticmethod
    def _obj_func(beta: np.ndarray, X: np.ndarray, y: np.ndarray, C: float):
        beta_x = X @ beta
        z = LogRidge._sigmoid(beta_x)
        logl = -np.sum(y * np.log(z + 1e-12) + (1 - y) * np.log(1 - z + 1e-12))
        penalty = (
            1.0 / (2.0 * C) * (np.sum(beta[1:] ** 2))
        )  # + np.sum(np.abs(beta[1:])))
        return logl + penalty

    @staticmethod
    def _gradient(beta: np.ndarray, X, y, C):
        z = LogRidge._sigmoid(X @ beta)
        grad = X.T @ (z - y) + (1.0 / C) * np.r_[0, *beta[1:]]
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(y)
        Xp = np.hstack((np.ones((n, 1)), X))

        args = (Xp, y, self.C)
        x0 = np.zeros(Xp.shape[1])
        res = minimize(
            fun=self._obj_func, jac=self._gradient, x0=x0, args=args, method="Newton-CG"
        )
        self.beta = res.x
        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray):
        n = X.shape[0]
        Xp = np.hstack((np.ones((n, 1)), X))
        proba = self._sigmoid(Xp @ self.beta)
        return np.vstack([1 - proba, proba]).T

    def predict(self, X: np.ndarray):
        return (self.predict_proba(X) >= 0.5).astype(int)


class KernelLogRidge(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel="rbf", **kernel_params):
        self.C = C
        self.kernel = kernel
        self.kernel_params = kernel_params

    @staticmethod
    def _sigmoid(z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def _obj_func(self, alpha, K, y):
        z = K @ alpha
        proba = self._sigmoid(z)
        loglik = -np.sum(
            y * np.log(proba + 1e-12) + (1 - y) * np.log(1 - proba + 1e-12)
        )
        penalty = (1 / (2 * self.C)) * np.sum(alpha**2)
        return loglik + penalty

    def _gradient(self, alpha, K, y):
        z = K @ alpha
        proba = self._sigmoid(z)
        grad = K.T @ (proba - y) + alpha / self.C
        return grad

    def fit(self, X, y):
        self.X_train_ = X
        K = pairwise_kernels(X, X, metric=self.kernel, **self.kernel_params)
        x0 = np.zeros(X.shape[0])
        res = minimize(
            fun=self._obj_func,
            jac=self._gradient,
            x0=x0,
            args=(K, y),
            method="L-BFGS-B",
        )
        self.alpha_ = res.x
        return self

    def predict_proba(self, X):
        if not hasattr(self, "alpha_"):
            raise NotFittedError("Call fit before predict.")
        K_test = pairwise_kernels(
            X, self.X_train_, metric=self.kernel, **self.kernel_params
        )
        proba = self._sigmoid(K_test @ self.alpha_)
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
