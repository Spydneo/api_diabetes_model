import pandas as pd
import numpy as np
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import os

def main ():
    # Uso de api de MLFlow con objeto cliente
    mlflow_client = mlflow.client.MlflowClient("http://127.0.0.1:5000")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("diabetes_experiment")
    # Almacenamiento de parametros y modelo en MLFlow
    mlflow.sklearn.autolog()

    # Obtención del Dataset
    os.system('ls')
    data = pd.read_csv('model/dataset/diabetes.csv')

    # Preprocesamiento Insulina
    data_filter = data[data['Insulin'] < 600]
    # Transformación de valores de insulina = 0 a Nan
    df_insuline_0 = data_filter[data_filter['Insulin'] == 0]
    df_insuline_0.loc[0:,4:5] = np.NaN
    df_insuline_no_0 = data_filter[data_filter['Insulin'] > 1]
    df_with_nan_insuline = pd.concat([df_insuline_no_0, df_insuline_0])
    df_with_nan_insuline.sort_index(inplace=True)

    # Obtención de X e y
    X = df_with_nan_insuline.drop(['Outcome'], axis=1)
    y = df_with_nan_insuline['Outcome']

    # Entrenamiento e imputación
    pipeline = Pipeline([
        ("KnnImputer", KNNImputer(weights='uniform')),
        ("model", KNeighborsClassifier())
    ])


    param_neighbors ={
        'KnnImputer__n_neighbors': list(range(1,10)),
        'model__n_neighbors': list(range(1,10))
    } 

    # Evaluación de los distintos parametros aplicando Cross Validation
    pipeline = RandomizedSearchCV(
                                    pipeline,
                                    param_neighbors,
                                    scoring = 'accuracy',
                                    verbose = True,
                                    n_iter = 20,
                                    cv= 9
        
    )

    pipeline.fit(X, y)

    mlflow.sklearn.log_model(pipeline, "model", registered_model_name='diabetes')

if __name__ == '__main__':
    main()



