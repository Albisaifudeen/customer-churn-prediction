import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os
from preprocess import load_and_clean, encode_and_split, BASE_DIR

def train():
    df = load_and_clean()
    X_train, X_test, y_train, y_test = encode_and_split(df)

    # MLflow 3.x requires SQLite backend
    db_path = os.path.join(BASE_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run():
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")

        print(classification_report(y_test, preds))
        print(f"ROC AUC: {auc:.4f}")

        models_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model, os.path.join(models_dir, "xgb_churn.pkl"))

if __name__ == "__main__":
    train()
