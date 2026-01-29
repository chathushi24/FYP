import numpy as np

try:
    import shap
except Exception:
    shap = None


def explain_rf(model, X: np.ndarray, top_k: int = 15):
    """
    Explain a RandomForest prediction for one sample X (shape (1, d)).
    Returns a dict that the Streamlit app can display.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != 1:
        raise ValueError("X must be shape (1, d)")

    # SHAP if available
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # multi-class returns list; take class 0 by default
            if isinstance(shap_values, list):
                sv = np.array(shap_values[0]).reshape(-1)
            else:
                sv = np.array(shap_values).reshape(-1)

            idx = np.argsort(np.abs(sv))[-top_k:][::-1]
            return {
                "method": "SHAP(TreeExplainer)",
                "top_features": idx.tolist(),
                "values": sv[idx].tolist()
            }
        except Exception:
            pass

    # Fallback: feature importance
    if hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_).reshape(-1)
        idx = np.argsort(imp)[-top_k:][::-1]
        return {
            "method": "feature_importances_ (fallback)",
            "top_features": idx.tolist(),
            "values": imp[idx].tolist()
        }

    return {
        "method": "none",
        "top_features": [],
        "values": []
    }
