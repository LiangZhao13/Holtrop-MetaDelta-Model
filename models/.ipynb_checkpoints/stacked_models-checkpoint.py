# models/stacked_models.py
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class TabNetRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for TabNetRegressor.

    IMPORTANT
    ---------
    This class MUST be importable before joblib.load(tabnet_model.pkl),
    otherwise unpickling will fail.
    """
    def __init__(
        self,
        max_epochs=100,
        patience=10,
        device_name="cpu",
        seed=42,
        verbose=0,
    ):
        self.max_epochs = max_epochs
        self.patience = patience
        self.device_name = device_name
        self.seed = seed
        self.verbose = verbose
        self.model = None
        self._estimator_type = "regressor"

    def fit(self, X, y):
        # TabNet expects y with shape (n_samples, 1)
        self.model = TabNetRegressor(
            device_name=self.device_name,
            seed=self.seed,
            verbose=self.verbose,
        )
        self.model.fit(
            X,
            y.reshape(-1, 1),
            max_epochs=self.max_epochs,
            patience=self.patience,
        )
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return preds.flatten()


class MetaNet(nn.Module):
    """
    Meta network for stacking-based ensemble learning.
    """
    def __init__(self, input_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))