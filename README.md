Creación y despliegue de un modelo de Inteligencia Artificial: 
=============================================================
Nota: En esta rama se ejecuta correctamente la creación de la Imagen de Docker pero no carga desde servidor de Mlflow.

Paquete que sirve para la inferencia de un modelo a través de una API Rest (Fast Api). En este caso se obtiene el modelo de en local (ltima versión diabetes). En la rama "feateure_mlflow_new_diabetes" se encuentra la ultima versión que hace uso de mlflow para cargar el modelo. También es posible crear una imagen de Docker haciendo un build del archivo Dockerfile.

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

Autor: Jairo Calderón