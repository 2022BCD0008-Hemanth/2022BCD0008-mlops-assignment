import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import joblib
import os
 

os.makedirs("models", exist_ok=True)
model_path = "models/model.pkl"

df = pd.read_csv("data/dataset.csv")

mlflow.set_experiment("2022BCD0008_experiment")

#5 runs
runs = [
    # Run 1 → Version 1, all features
    ("run1", LogisticRegression(max_iter=200), df.columns[:-1]),

    # Run 2 → Hyperparameter change
    ("run2", LogisticRegression(C=0.5, max_iter=200), df.columns[:-1]),

    # Run 3 → Same model, assume better dataset
    ("run3", LogisticRegression(max_iter=300), df.columns[:-1]),

    # Run 4 → Feature selection 
    ("run4", LogisticRegression(max_iter=200), df.columns[:2]),

    # Run 5 → Different model + feature selection
    ("run5", RandomForestClassifier(n_estimators=100), df.columns[:-1]),
] 
os.makedirs("models", exist_ok=True)
 
for name, model, features in runs:
    with mlflow.start_run(run_name=name):

        X = df[list(features)]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        # Log parameters
        mlflow.log_param("model", str(model))
        mlflow.log_param("features_used", list(features))

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Save model
        model_path = f"models/{name}_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"{name}: Accuracy={acc}, F1={f1}")
