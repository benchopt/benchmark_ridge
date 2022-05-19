from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'GD'

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}
    support_sparse = False

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - not implemented but this should be quite easy
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept=False):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        L = np.linalg.norm(self.X, ord=2) ** 2
        n_features = self.X.shape[1]
        w = np.zeros(n_features)
        w_acc = np.zeros(n_features)
        w_old = np.zeros(n_features)
        t_new = 1
        for _ in range(n_iter):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                w_old[:] = w  # x in Beck & Teboulle (2009) notation
                w[:] = w_acc  # y in Beck & Teboulle (2009) notation
            w -= 1 / L * (
                self.X.T.dot(self.X.dot(w) - self.y) + self.lmbd * w
                )
            if self.use_acceleration:
                w_acc[:] = w + (t_old - 1.) / t_new * (w - w_old)
        self.w = w

    def get_result(self):
        return self.w
