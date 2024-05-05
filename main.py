from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import keras

model = keras.models.load_model('breast_cancer.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

class Observation(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float



app = FastAPI()

@app.get('/')
def index():
    return {"hellow": "world"}

@app.post('/predict')
async def predict(observation: Observation):
    array = np.asarray(list(observation.model_dump().values())).reshape(1, -1)
    array_scld = scaler.transform(array)
    prediction = model.predict(array_scld)
    prediction_label = [0 if y[0] < 0.5 else 1 for y in prediction]

    if(prediction_label[0] == 0):
       return {'prediction': 0, 'message': 'The tumor is Malignant'}

    else:
       return {'prediction': 1, 'message': 'The tumor is benign'}
