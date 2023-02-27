import pickle
from pathlib import Path
import numpy as np
import pandas as pd

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


classes = [
    "setosa",
    "versicolor",
    "virginica"
]


def predict_pipeline(measurements):
    measurements = pd.DataFrame(np.array([measurements]), columns=["sepal length (cm)",  "sepal width (cm)",  "petal length (cm)",  "petal width (cm)"])
    pred = model.predict(measurements)
    return classes[pred[0]]