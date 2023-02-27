from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version


app = FastAPI()


class MeasurementsIn(BaseModel):
    measurements: str


class PredictionOut(BaseModel):
    predicted_species: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: MeasurementsIn):
    measurements = payload.measurements.split(",")
    measurements = [float(x) for x in measurements]
    predicted_species = predict_pipeline(measurements)
    return {"predicted_species": predicted_species}