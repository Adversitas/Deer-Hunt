"""L2-regularised logistic regression in numpy, with built-in feature scaling.

Kept deliberately simple: financial returns are noisy and a linear model is
far less prone to overfitting than a deep net on a few thousand bars.
"""

import numpy as np


class LogisticModel:
    def __init__(self, l2=0.01, lr=0.1, epochs=500):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.mean = None
        self.std = None

    def _scale(self, X):
        return np.clip((X - self.mean) / self.std, -5, 5)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Xs = self._scale(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))
        for _ in range(self.epochs):
            p = _sigmoid(Xs @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (Xs.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return _sigmoid(self._scale(np.asarray(X, dtype=float)) @ self.w + self.b)


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
