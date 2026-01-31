# inference/radded_predictor.py
import os
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# VERY IMPORTANT:
# This import guarantees TabNetRegressorWrapper is visible during unpickling
from models.stacked_models import TabNetRegressorWrapper, MetaNet  # noqa: F401


def predict_r_added_from_dir(
    df,
    model_dir,
    feature_cols,
    device="cpu",
    save_scaler=True,
    scaler_name="scaler.pkl",
    filenames=None,
):
    """
    Predict R_added using stacked base models loaded from a directory.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing feature columns.
    model_dir : str
        Directory containing trained models.
    feature_cols : list[str]
        Feature columns used for prediction.
    device : str, optional
        Torch device, default "cpu".
    save_scaler : bool, optional
        Save fitted scaler if no scaler file exists.
    scaler_name : str, optional
        Scaler filename inside model_dir.
    filenames : dict or None
        Optional custom filenames for models.

    Returns
    -------
    r_added : np.ndarray
        Predicted R_added (n_samples,).
    df_out : pandas.DataFrame
        Copy of df with column 'R_a_pre'.
    scaler : StandardScaler
        Scaler used during inference.
    """

    # --------------------------------------------------
    # 0. Default filenames
    # --------------------------------------------------
    if filenames is None:
        filenames = {
            "xgb": "xgb_model.pkl",
            "cat": "catboost_model.pkl",
            "tabnet": "tabnet_model_fixed.pkl",
            "tabpfn": "tabpfn_model.pkl",
            "meta": "meta_model.pth",
        }

    # --------------------------------------------------
    # 1. Load base models
    # --------------------------------------------------
    xgb_model = joblib.load(os.path.join(model_dir, filenames["xgb"]))
    catboost_model = joblib.load(os.path.join(model_dir, filenames["cat"]))
    tabnet_model = joblib.load(os.path.join(model_dir, filenames["tabnet"]))
    tabpfn_model = joblib.load(os.path.join(model_dir, filenames["tabpfn"]))

    # --------------------------------------------------
    # 2. Load MetaNet
    # --------------------------------------------------
    meta_model = MetaNet(input_dim=4).to(device)
    meta_model.load_state_dict(
        torch.load(os.path.join(model_dir, filenames["meta"]), map_location=device)
    )
    meta_model.eval()

    # --------------------------------------------------
    # 3. Feature extraction & scaling
    # --------------------------------------------------
    X = df[feature_cols].values
    scaler_path = os.path.join(model_dir, scaler_name)

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if save_scaler:
            joblib.dump(scaler, scaler_path)

    # --------------------------------------------------
    # 4. Base model predictions
    # --------------------------------------------------
    preds_xgb = xgb_model.predict(X_scaled)
    preds_cat = catboost_model.predict(X_scaled)
    preds_tabnet = tabnet_model.predict(X_scaled)
    preds_tabpfn = tabpfn_model.predict(X_scaled)

    stacked_preds = np.column_stack(
        [preds_xgb, preds_cat, preds_tabnet, preds_tabpfn]
    )

    # --------------------------------------------------
    # 5. MetaNet fusion
    # --------------------------------------------------
    X_meta = torch.tensor(stacked_preds, dtype=torch.float32, device=device)

    with torch.no_grad():
        r_added = meta_model(X_meta).cpu().numpy().flatten()

    df_out = df.copy()
    df_out["R_a_pre"] = r_added

    return r_added, df_out, scaler