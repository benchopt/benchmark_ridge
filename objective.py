import numpy as np

from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ridge Regression"

    parameters = {
        "fit_intercept": [True, False],
        'lmbd': [0.1]
    }

    def __init__(self, lmbd=0.1, fit_intercept=False):
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept

    def set_data(self, X, y, X_test=None, y_test=None):
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test
        self.n_features = self.X.shape[1]

    def get_one_solution(self):
        n_features = self.n_features
        if self.fit_intercept:
            n_features += 1
        return np.zeros(n_features)

    def compute(self, beta):
        # compute residuals
        test_loss = None
        if self.X_test is not None:
            test_loss = self._compute_loss(
                self.X_test, self.y_test, self.lmbd, beta
            )
        train_loss = self._compute_loss(self.X, self.y, self.lmbd, beta)
        return {"value": train_loss, "Test loss": test_loss}

    def _compute_loss(self, X, y, lmbd, beta):
        if self.fit_intercept:
            beta, intercept = beta[: self.n_features], beta[self.n_features:]
        diff = y - X @ beta
        if self.fit_intercept:
            diff -= intercept
        return 0.5 * diff.dot(diff) + lmbd * 0.5 * beta @ beta

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            lmbd=self.lmbd,
            fit_intercept=self.fit_intercept,
        )
