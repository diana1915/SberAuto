from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from catboost import *
import dill


app = FastAPI()
with open('visit.pkl', 'rb') as file:
    model = dill.load(file)

with open('preprocessor.pkl', 'rb') as file:
    prepro = dill.load(file)


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    pred: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    df = prepro['preprocessor'].fit_transform(df)

    cat_features = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

    pool = Pool(data=df, cat_features=cat_features)
    y = model['model'].predict(pool)

    return {
        'pred': y[0],
    }
