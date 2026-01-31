# ==================================================
# R_added ensemble model training script
# ==================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor  # kept for compatibility, not used
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
import joblib

# --------------------------------------------------
# Import TabNet, TabPFN, and sklearn base classes
# --------------------------------------------------
from pytorch_tabnet.tab_model import TabNetRegressor
from tabpfn import TabPFNRegressor
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Wrapper for TabNet to make it compatible with sklearn
# --------------------------------------------------
class TabNetRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, max_epochs=100, patience=10, device_name='cpu', seed=42, verbose=0):
        self.max_epochs = max_epochs
        self.patience = patience
        self.device_name = device_name
        self.seed = seed
        self.verbose = verbose
        self.model = None
        # Explicitly declare this estimator as a regressor
        self._estimator_type = "regressor"

    def fit(self, X, y):
        # TabNet requires y to have shape (n_samples, 1)
        self.model = TabNetRegressor(
            device_name=self.device_name,
            seed=self.seed,
            verbose=self.verbose
        )
        self.model.fit(
            X,
            y.reshape(-1, 1),
            max_epochs=self.max_epochs,
            patience=self.patience
        )
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return preds.flatten()

# --------------------------------------------------
# Data loading and preprocessing
# --------------------------------------------------
data = pd.read_csv('/kaggle/input/mrvdatase/data_13_17_1_std.csv')  # replace with actual path if needed

# Drop unused columns
data = data.drop(
    ['Unnamed: 0', 'Pe', 'R_total', 'R_calm', 'postime', 'num'],
    axis=1
)

# --------------------------------------------------
# Outlier detection using KMeans clustering
# --------------------------------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Assume the smaller cluster corresponds to anomalies and remove it
cluster_sizes = data['cluster'].value_counts()
anomaly_cluster = cluster_sizes.idxmin()
data = data[data['cluster'] != anomaly_cluster]
data = data.drop(columns=['cluster'])

# --------------------------------------------------
# Feature-target separation and normalization
# --------------------------------------------------
features = data.drop(columns=['R_added']).values
targets = data['R_added'].values

# Standardize features
features = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Optuna objective function for XGBoost
# --------------------------------------------------
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **params
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

# --------------------------------------------------
# Optuna objective function for CatBoost
# --------------------------------------------------
def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 1)
    }

    model = CatBoostRegressor(
        verbose=0,
        random_seed=42,
        **params
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=10,
        verbose=False
    )

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

# --------------------------------------------------
# Hyperparameter optimization with Optuna
# --------------------------------------------------
study_xgb = optuna.create_study(direction='minimize', study_name='XGBoost Optimization')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)

study_catboost = optuna.create_study(direction='minimize', study_name='CatBoost Optimization')
study_catboost.optimize(objective_catboost, n_trials=50, show_progress_bar=False)

print("XGBoost Best parameters:", study_xgb.best_params)
print("XGBoost Best RMSE:", study_xgb.best_value)
print("CatBoost Best parameters:", study_catboost.best_params)
print("CatBoost Best RMSE:", study_catboost.best_value)

# --------------------------------------------------
# Define base models
# --------------------------------------------------
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    **study_xgb.best_params
)

catboost_model = CatBoostRegressor(
    verbose=0,
    random_seed=42,
    **study_catboost.best_params
)

tabnet_model = TabNetRegressorWrapper(
    max_epochs=300,
    patience=10,
    device_name='cuda',
    seed=42,
    verbose=0
)

# If GPU is available, set device='cuda'
tabpfn_model = TabPFNRegressor(
    device='cuda',
    ignore_pretraining_limits=True
)

# --------------------------------------------------
# Define meta-model (neural network)
# --------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

class MetaNet(nn.Module):
    def __init__(self, input_dim):
        super(MetaNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --------------------------------------------------
# Neural stacking regressor
# --------------------------------------------------
class NeuralStackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models):
        self.base_models = base_models
        self.meta_model = None

    def fit(self, X, y):
        # Train base models
        for name, model in self.base_models:
            model.fit(X, y)

        # Generate stacked features
        preds = []
        for name, model in self.base_models:
            preds.append(model.predict(X))
        stacked_train = np.column_stack(preds)

        # Train meta-model
        self.meta_model = MetaNet(input_dim=stacked_train.shape[1])
        X_meta = torch.tensor(stacked_train, dtype=torch.float32)
        y_meta = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.meta_model.parameters(), lr=0.01)

        num_epochs = 400
        for epoch in range(num_epochs):
            self.meta_model.train()
            optimizer.zero_grad()
            outputs = self.meta_model(X_meta)
            loss = criterion(outputs, y_meta)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Meta-model Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        return self

    def predict(self, X):
        preds = []
        for name, model in self.base_models:
            preds.append(model.predict(X))
        stacked_test = np.column_stack(preds)

        self.meta_model.eval()
        with torch.no_grad():
            X_meta = torch.tensor(stacked_test, dtype=torch.float32)
            meta_preds = self.meta_model(X_meta).numpy().flatten()

        return meta_preds

# --------------------------------------------------
# Train ensemble model and evaluate
# --------------------------------------------------
base_models = [
    ('xgb', xgb_model),
    ('catboost', catboost_model),
    ('tabnet', tabnet_model),
    ('tabpfn', tabpfn_model)
]

ensemble_model = NeuralStackingRegressor(base_models=base_models)
ensemble_model.fit(X_train, y_train)

y_pred = ensemble_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# --------------------------------------------------
# Save trained models
# --------------------------------------------------
torch.save(ensemble_model.meta_model.state_dict(), "meta_model.pth")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(catboost_model, "catboost_model.pkl")
joblib.dump(tabnet_model, "tabnet_model.pkl")
joblib.dump(tabpfn_model, "tabpfn_model.pkl")