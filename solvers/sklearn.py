import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'sklearn'

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    references = [
        'F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, '
        'O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, '
        'J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot'
        ' and E. Duchesnay'
        '"Scikit-learn: Machine Learning in Python", J. Mach. Learn. Res., '
        'vol. 12, pp. 2825-283 (2011)'
    ]
    parameters = {
        "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "saga"],
    }

    def __init__(self, solver="svd"):
        self.solver = solver

    def set_objective(self, X, y, lmbd=1, fit_intercept=False):
        self.X, self.y, self.fit_intercept = X, y, fit_intercept
        self.ridge = Ridge(
            fit_intercept=fit_intercept, alpha=lmbd, solver=self.solver,
            tol=1e-10)

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1] + self.fit_intercept])
        else:
            self.ridge.max_iter = n_iter
            self.ridge.fit(self.X, self.y)
            coef = self.ridge.coef_.flatten()
            if self.fit_intercept:
                coef = np.r_[coef, self.ridge.intercept_]
            self.coef = coef

    def get_result(self):
        return self.coef
