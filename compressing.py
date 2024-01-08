#!/usr/bin/env python3

import pandas as pd

df = pd.read_csv("adult-data.csv")
print(df.shape)
df.drop(['education'], axis=1, inplace=True)
df.drop(['marital-status'], axis=1, inplace=True)

for ind in df.index:
    if df['relationship'][ind] == 'Wife' or df['relationship'][ind] == 'Husband':
        df['relationship'][ind] == 1
    else:
        df['relationship'][ind] == 0

print(df.shape)

df.to_csv("lighter-data.csv")