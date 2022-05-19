from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import get_cuda_version

cuda_version = get_cuda_version()
if cuda_version is not None:
    cuda_version = cuda_version.split("cuda_", 1)[1][:4]

with safe_import_context() as import_ctx:
    if cuda_version is None:
        raise ImportError("cuml solver needs a nvidia GPU.")
    from scipy import sparse
    import cudf
    import numpy as np
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    from cuml.linear_model import Ridge

# XXX TODO: add solver as parameter


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "rapidsai::rapids",
        f"nvidia::cudatoolkit={cuda_version}",
        "dask-sql",
    ] if cuda_version is not None else []

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        if sparse.issparse(X):
            if sparse.isspmatrix_csc(X):
                self.X = cusparse.csc_matrix(X)
            elif sparse.isspmatrix_csr(X):
                self.X = cusparse.csr_matrix(X)
            else:
                raise ValueError("Non suported sparse format")
        else:
            self.X = cudf.DataFrame(self.X.astype(np.float32))
        self.y = cudf.Series(self.y)
        self.fit_intercept = fit_intercept

        self.ridge = Ridge(
            fit_intercept=self.fit_intercept,
            alpha=self.lmbd / self.X.shape[0],
            verbose=0,
        )

    def run(self, n_iter):
        self.ridge.max_iter = n_iter
        self.ridge.fit(self.X, self.y)

    def get_result(self):
        if isinstance(self.ridge.coef_, cp.ndarray):
            coef = self.ridge.coef_.get().flatten()
            if self.ridge.fit_intercept:
                coef = np.r_[coef, self.ridge.intercept_.get()]
        else:
            coef = self.ridge.coef_.to_numpy().flatten()
            if self.ridge.fit_intercept:
                coef = np.r_[coef, self.ridge.intercept_]

        return coef.astype(np.float64)
