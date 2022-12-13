import joblib
import os
from sklearn import svm
from sklearn import datasets
import mlflow

# Uso de api de MLFlow con objeto cliente
mlflow_client = mlflow.client.MlflowClient("http://localhost:5000")
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris_experiment")


def load_dataset():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    return X, y

def train(X, y):
    # Almacenamiento de parametros y modelo en MLFlow
    mlflow.sklearn.autolog()
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    return clf

def save_model(clf):
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/iris_classifier.joblib")


if __name__  == "__main__":
    X, y = load_dataset()
    clf = train(X, y)
    # save_model(clf)
    mlflow.sklearn.log_model(clf, "model", registered_model_name='iris')