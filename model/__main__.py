import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
import os

def main ():
    # Obtención del Dataset
    os.system('ls')
    data = pd.read_csv('model/dataset/diabetes.csv')
    # Creando un df con las columnas con medidas médicas con valores igual cero
    data_columns_0 = data.drop(["Pregnancies", "DiabetesPedigreeFunction", "Age", "Outcome"], axis='columns') 
    
    # Imputación
    data_columns_impute = pd.DataFrame()
    # Distibución sesgada hacia la izquierda por lo que se utiliza la mediana para imputar los valores
    data_columns_impute["Insulin"] = data_columns_0["Insulin"].replace(0, data_columns_0["Insulin"].median())
    # Distibución sesgada hacia la izquierda por lo que se utiliza la mediana para imputar los valores
    data_columns_impute["SkinThickness"] = data_columns_0["SkinThickness"].replace(0, data_columns_0["SkinThickness"].median())
    # Distibución normal o centrada por lo que se utiliza la media para imputar los valores
    data_columns_impute["BloodPressure"] = data_columns_0["BloodPressure"].replace(0, data_columns_0["BloodPressure"].mean())
    # Distibución normal o centrada por lo que se utiliza la media para imputar los valores
    data_columns_impute["BMI"] = data_columns_0["BMI"].replace(0, data_columns_0["BMI"].mean())
    # Distibución normal o centrada por lo que se utiliza la media para imputar los valores
    data_columns_impute["Glucose"] = data_columns_0["Glucose"].replace(0, data_columns_0["Glucose"].mean())

    # Creamos el nuevo Dataframe con los valores imputados 
    data_drop = data.drop(columns= data_columns_0.keys()) # Dataframe con columnas no imputadas
    # Dataframe completo con columnas imputadas
    data_imputed = pd.concat([data_drop, data_columns_impute], axis=1)

    # Separación en X e y
    X = data_imputed.drop(['Outcome'], axis=1)
    y = data_imputed['Outcome']

    # Normalización
    quantile_scale = QuantileTransformer()
    data_scaled = quantile_scale.fit_transform(X)
    data_scaled = pd.DataFrame(data_scaled)
    data_scaled.columns =["Pregnancies",	"DiabetesPedigreeFunction",	"Age",	"Insulin",	"SkinThickness",	"BloodPressure",	"BMI",	"Glucose"]

    # Actualizo valor de X con df escalado
    X = data_scaled

    # Validación Cruzada módificada
    cv = RepeatedStratifiedKFold(n_splits=15, n_repeats=5, random_state=456)

    # Creamos Pipeline
    pipeline = Pipeline([
        ("model", KNeighborsClassifier())
    ])

    # Valores de n_neighbors a ser probados
    param_neighbors ={
        'model__n_neighbors': list(range(1,100))
    } 

    # Evaluación de los distintos parametros aplicando un Cross Validation modificado (RepeatedStratifiedKFold)
    pipeline = RandomizedSearchCV(
                                    pipeline,
                                    param_neighbors,
                                    scoring = 'accuracy',
                                    verbose = True,
                                    n_iter = 200,
                                    cv= cv
        
    )

    pipeline.fit(X, y)

    with open('model/pickle/diabetes_model.pkl', 'wb') as f:
        pickle.dump(pipeline.best_estimator_, f)

if __name__ == '__main__':
    main()