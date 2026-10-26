from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Flower Prediction API")

model_path = Path("models/model.joblib")
model = joblib.load(model_path)

class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def health_check():
    return {"message": "Welcome to the Iris Flower Prediction API!"}

@app.post("/predict")
def predict_iris_flower(data: IrisRequest):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)
    return {
        "prediction": int(prediction[0]),
        "class_name": ["setosa", "versicolor", "virginica"][int(prediction[0])]
    }


class InputData(BaseModel):
    sepal_length: float = Field(description="Sepal length")
    sepal_width: float = Field(description="Sepal width")
    petal_length: float = Field(description="Petal length")
    petal_width: float = Field(description="Petal width")


class PredictionResult(BaseModel):
    prediction: int