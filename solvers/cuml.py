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
    import cupyx.scipy.sparse as cusparse
    from cuml.linear_model import Ridge


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "rapidsai::rapids",
        f"nvidia::cudatoolkit={cuda_version}",
        "dask-sql",
    ] if cuda_version is not None else []

    parameters = {
        "solver": [
            "eig",
            "svd",
            "cd",
        ],
    }

    parameter_template = "{solver}"

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
            solver=self.solver,
            verbose=0,
        )

    def run(self, n_iter):
        self.ridge.solver_model.max_iter = n_iter
        self.ridge.fit(self.X, self.y)

    def get_result(self):
        return self.ridge.coef_.to_numpy().flatten()
