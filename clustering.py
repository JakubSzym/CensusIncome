#!/usr/bin/env python3

import pandas as pd
from pycaret.clustering import setup, create_model, predict_model

df = pd.read_csv("lighter-data.csv")

s = setup(df)

model = create_model('kmeans')

predictions = predict_model(model, data=df)

print(predictions)