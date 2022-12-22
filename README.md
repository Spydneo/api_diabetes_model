Creación y despliegue de un modelo de Inteligencia Artificial: 
=============================================================
Paquete que sirve para la inferencia de un modelo a través de una API Rest (Fast Api). En este caso se obtiene el modelo de un servidor de MlFlow que sirva para almacenar y versionar los modelos entrenados. También es posible crear una imagen de Docker haciendo un build del archivo Dockerfile.

Para entrenar el modelo podemos ejecutar:
```bash
 python -m modelo
```

Para arrancar el servicio de Fast API:
```bash
uvicorn api_diabetes_model.main:app --port 5000 --reload
```

Para arrancar el servicio de MlFlow:
```bash
mlflow ui --port 5000 --serve-artifacts --backend-store-uri sqlite:///mlflowdb.sqlite
```

Para crear la imagen de Docker:
```bash
docker build -t diabetes_api .
```

Se puede comentar el apartado de MLFlow y descomentar la carga del modelo en local (pickle) si no se quiere utilizar mlflow.

Autor: Jairo Calderón
