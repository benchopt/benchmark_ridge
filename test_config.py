import sys
import pytest
from benchopt.utils.sys_info import get_cuda_version


def check_test_solver_install(solver_class):
    if "cuml" in solver_class.name.lower():
        if sys.platform == "darwin":
            pytest.xfail("Cuml is not supported on MacOS.")
        cuda_version = get_cuda_version()
        if cuda_version is None:
            pytest.xfail("Cuml needs a working GPU hardware.")

    if "glmnet" in solver_class.name.lower():
        pytest.xfail("glmnet produces discrepancies (see issue #2).")
