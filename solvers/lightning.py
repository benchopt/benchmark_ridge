from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from lightning.regression import CDRegressor


class Solver(BaseSolver):
    name = 'Lightning'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/scikit-learn-contrib/lightning.git'
    ]

    references = [
        'M. Blondel, K. Seki and K. Uehara, '
        '"Block coordinate descent algorithms for large-scale sparse '
        'multiclass classification" '
        'Mach. Learn., vol. 93, no. 1, pp. 31-52 (2013)'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        # lightning does not handle intercept properly (always set to zero)
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):

        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept
        self.ridge = CDRegressor(
            loss='squared', penalty='l2', C=.5, alpha=self.lmbd,
            tol=0, permute=False, shrinking=False, warm_start=False)

    def run(self, n_iter):
        self.ridge.max_iter = n_iter
        self.ridge.fit(self.X, self.y)

    def get_result(self):
        beta = self.ridge.coef_.flatten()
        if self.fit_intercept:
            beta = np.r_[beta, self.ridge.intercept_]
        return beta
