# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import pickle

# Loading the Saved Model
loaded_model = pickle.load(open('D:/Breast Cancer Detection/Models/ML on csv data/ML_csv_trained_model.sav','rb'))

new_data = {
    'id': 1000,
    'radius_mean': 12.5,
    'texture_mean': 18.0,
    'perimeter_mean': 78.0,
    'area_mean': 500.0,
    'smoothness_mean': 0.1,
    'compactness_mean': 0.08,
    'concavity_mean': 0.06,
    'concave points_mean': 0.03,
    'symmetry_mean': 0.2,
    'fractal_dimension_mean': 0.05,
    'radius_se': 0.3,
    'texture_se': 1.2,
    'perimeter_se': 2.5,
    'area_se': 30.0,
    'smoothness_se': 0.005,
    'compactness_se': 0.02,
    'concavity_se': 0.015,
    'concave points_se': 0.008,
    'symmetry_se': 0.015,
    'fractal_dimension_se': 0.002,
    'radius_worst': 15.0,
    'texture_worst': 22.0,
    'perimeter_worst': 90.0,
    'area_worst': 600.0,
    'smoothness_worst': 0.12,
    'compactness_worst': 0.25,
    'concavity_worst': 0.2,
    'concave points_worst': 0.1,
    'symmetry_worst': 0.28,
    'fractal_dimension_worst': 0.07
}

from sklearn.impute import SimpleImputer
# Convert the new data to a DataFrame
new_data_df = pd.DataFrame([new_data])

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
new_data_df_imputed = pd.DataFrame(imputer.fit_transform(new_data_df), columns=new_data_df.columns)

# Make predictions on the new data
new_data_predictions = loaded_model.predict(new_data_df_imputed)

if new_data_predictions[0] == 1:
    print("Prediction: The patient is predicted to have cancer.")
else:
    print("Prediction: The patient is predicted to be cancer-free.")
