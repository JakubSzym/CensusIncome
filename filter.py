#!/usr/bin/env python3

import csv

filtered_data = []
field_names = ['age', 'workclass','fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'income']
missing_values, all_values = 0, 0
with open('adult-original-data.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    clean_row = { k:v.strip() for k, v in row.items()}
    if not '?' in clean_row.values():
      filtered_data.append(clean_row)
      all_values += 1
    else:
      missing_values += 1

with open('adult-data.csv', 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=field_names)
  writer.writeheader()
  for row in filtered_data:
    writer.writerow(row)

print(f"Filtration done, deleted {missing_values} missing values.")
print(f"Dataset contains {all_values} items after filtration")


