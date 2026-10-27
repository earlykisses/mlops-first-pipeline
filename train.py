from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:6001")  # Set the MLflow tracking URI

mlflow.set_experiment("iris-classification")  # Set the experiment name

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42
)

n_estimators = 100
random_state = 42

# Model settings we can log to MLflow
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.2
}

with mlflow.start_run() as run:
    mlflow.log_params(model_parameters)  # Log model parameters to MLflow

    # Initialize the Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)  # Train the model

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # Log the metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

print("MLflow run ID:", run.info.run_id)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Save the trained model
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "model.joblib")
print("Model saved to 'models/model.joblib'")


# Log the trained model to MLflow
model_info = mlflow.sklearn.log_model(
    sk_model=model,
    name="iris_model",
    skops_trusted_types={
        "sklearn.tree._tree.Tree"
    }
)

# Register the logged model
mlflow.register_model(
    model_uri=model_info.model_uri,
    name="iris_classifier"
)

print("Registered model: 'iris_classifier'")
