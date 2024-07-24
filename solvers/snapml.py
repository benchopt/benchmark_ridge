from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import get_cuda_version


with safe_import_context() as import_ctx:
    from snapml import LinearRegression
    import numpy as np


class Solver(BaseSolver):
    name = "snapml"

    install_cmd = "conda"

    # Snap ML wheels are built with numpy 1.*, they are not compatible with
    # numpy>=2. TODO : upgrade numpy when compatible wheels are released.

    # libomp x86_64 and Homebrew are needed for snapML to work correctly on
    # MacOS. Please check this tutorial to install SnapML correctly:
    # https://github.com/benchopt/benchmark_ridge/blob/tutorials/snapml_macos.rst
    requirements = ["numpy'<2'", "pip::snapml"]

    parameters = {"gpu": [False, True]}
    references = [
        "C. Dünner, T. Parnell, D. Sarigiannis, N. Ioannou, A. Anghel, "
        "G. Ravi, M. Kandasamy and H. Pozidis, "
        "'Snap ML: A hierarchical framework for machine learning', "
        "Advances in Neural Information Processing Systems (2018)"
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if self.gpu and get_cuda_version() is None:
            return True, "snapml[gpu=True] needs a GPU to run"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        self.ridge = LinearRegression(
            fit_intercept=self.fit_intercept,
            regularizer=self.lmbd,
            penalty="l2",
            tol=1e-12,
            dual=False,
            use_gpu=self.gpu,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1] + self.fit_intercept])
            return

        self.ridge.max_iter = n_iter
        self.ridge.fit(self.X, self.y)
        coef = self.ridge.coef_.flatten()
        if self.fit_intercept:
            coef = np.r_[coef, self.ridge.intercept_]
        self.coef = coef

    def get_result(self):
        return dict(beta=self.coef)
