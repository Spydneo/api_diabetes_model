from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import mlflow

app = FastAPI()

# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:8080",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##################### Load Model #########################
# Load model production version of MLFlow server
mlflow.set_tracking_uri("http://localhost:8000")
def load_model():
    os.system('ls')
    return mlflow.sklearn.load_model("models:/diabetes/production")

# def load_model():
#     os.system('ls')
    # return joblib.load("api_diabetes_model/model/iris_classifier.joblib")
    # return joblib.load("model/api_diabetes_model.joblib") #To Dockerfile Workdir

##################### BaseModel ###########################
class InferenceParameters (BaseModel):
    """Medidas de las carecteristicas de salud separadas por coma

    Args:
        medidas (string): String de medidas
    
    Ejemplo
    =======
    11,143,94,33,146,36.6,0.254,51
    """
    medidas: list

####################### EndPoints ##########################

@app.post("/predict/")
async def inference(input: InferenceParameters):
    clf = load_model()
    df_input = pd.DataFrame([input.medidas])
    result = clf.predict(df_input)
    
    return{"input": input, "result": float(result[0]) }

@app.get("/")
async def root():
    model = load_model()
    return {"message": "OK"}