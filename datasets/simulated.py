from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (100, 50),
            (1000, 2000)]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        beta = rng.randn(self.n_features)

        X = rng.randn(self.n_samples, self.n_features)
        y = X @ beta

        X_test = rng.randn(self.n_samples, self.n_features)
        y_test = X_test @ beta

        data = dict(X=X, y=y, X_test=X_test, y_test=y_test)

        return data
