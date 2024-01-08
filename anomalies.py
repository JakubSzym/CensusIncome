#!/usr/bin/env python3
import pandas as pd
from pycaret.anomaly import setup, create_model, assign_model, evaluate_model, predict_model, save_model, load_model
from IPython.display import display

df = pd.read_csv("lighter-data.csv")

s = setup(df)

model = create_model('knn')
predictions = predict_model(model, data=df)

predictions = predictions[predictions['Anomaly'] != 0]
display(predictions)
predictions.to_csv("anomalies.csv")

save_model(model, 'knn')