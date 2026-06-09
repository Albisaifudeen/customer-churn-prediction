import shap
import joblib
import os
import matplotlib.pyplot as plt
from preprocess import load_and_clean, encode_and_split, BASE_DIR

def explain():
    df = load_and_clean()
    X_train, X_test, _, _ = encode_and_split(df)

    model_path = os.path.join(BASE_DIR, "models", "xgb_churn.pkl")
    model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    figures_dir = os.path.join(BASE_DIR, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(figures_dir, "shap_summary.png"), bbox_inches="tight")
    print("SHAP plot saved to reports/figures/shap_summary.png")

if __name__ == "__main__":
    explain()