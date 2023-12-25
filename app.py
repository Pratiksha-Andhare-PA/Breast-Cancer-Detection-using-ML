
import pickle
import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the saved model
loaded_model = pickle.load(open('D:/Breast Cancer Detection/Models/ML on csv data/ML_csv_trained_model.sav', 'rb'))

# Function for prediction
def breast_cancer_prediction(input_data):
    print("Input data : ", input_data)
    # Convert input data to numerical values and create DataFrame
    new_data_df = pd.DataFrame([[
        float(input_data['p_id']), float(input_data['radius_mean']), float(input_data['texture_mean']),
        float(input_data['perimeter_mean']), float(input_data['area_mean']), float(input_data['smoothness_mean']),
        float(input_data['compactness_mean']), float(input_data['concavity_mean']),
        float(input_data['concave_points_mean']), float(input_data['symmetry_mean']),
        float(input_data['fractal_dimension_mean']), float(input_data['radius_se']), float(input_data['texture_se']),
        float(input_data['perimeter_se']), float(input_data['area_se']), float(input_data['smoothness_se']),
        float(input_data['compactness_se']), float(input_data['concavity_se']),
        float(input_data['concave_points_se']), float(input_data['symmetry_se']),
        float(input_data['fractal_dimension_se']), float(input_data['radius_worst']),
        float(input_data['texture_worst']), float(input_data['perimeter_worst']),
        float(input_data['area_worst']), float(input_data['smoothness_worst']),
        float(input_data['compactness_worst']), float(input_data['concavity_worst']),
        float(input_data['concave_points_worst']), float(input_data['symmetry_worst']),
        float(input_data['fractal_dimension_worst'])
    ]])

    print("dataframe : ", new_data_df)
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    new_data_imputed = imputer.fit_transform(new_data_df)

    # Make predictions
    new_data_predictions = loaded_model.predict(new_data_imputed)

    return "Prediction: The patient is predicted to have cancer." if new_data_predictions[0] == 1 else "Prediction: The patient is predicted to be cancer-free."

def main():
    st.title('Breast Cancer Prediction Web App')

    # Get input data from user
    new_data = {
        'p_id': st.text_input('Patient ID'),
        'radius_mean': st.number_input('Radius Mean'),
        'texture_mean': st.number_input('Texture Mean'),
        'perimeter_mean': st.number_input('Perimeter Mean'),
        'area_mean': st.number_input('Area Mean'),
        'smoothness_mean': st.number_input('Smoothness Mean'),
        'compactness_mean': st.number_input('Compactness Mean'),
        'concavity_mean': st.number_input('Concavity Mean'),
        'concave_points_mean': st.number_input('Concave Points Mean'),
        'symmetry_mean': st.number_input('Symmetry Mean'),
        'fractal_dimension_mean': st.number_input('Fractal Dimension Mean'),
        'radius_se': st.number_input('Radius SE'),
        'texture_se': st.number_input('Texture SE'),
        'perimeter_se': st.number_input('Perimeter SE'),
        'area_se': st.number_input('Area SE'),
        'smoothness_se': st.number_input('Smoothness SE'),
        'compactness_se': st.number_input('Compactness SE'),
        'concavity_se': st.number_input('Concavity SE'),
        'concave_points_se': st.number_input('Concave Points SE'),
        'symmetry_se': st.number_input('Symmetry SE'),
        'fractal_dimension_se': st.number_input('Fractal Dimension SE'),
        'radius_worst': st.number_input('Radius Worst'),
        'texture_worst': st.number_input('Texture Worst'),
        'perimeter_worst': st.number_input('Perimeter Worst'),
        'area_worst': st.number_input('Area Worst'),
        'smoothness_worst': st.number_input('Smoothness Worst'),
        'compactness_worst': st.number_input('Compactness Worst'),
        'concavity_worst': st.number_input('Concavity Worst'),
        'concave_points_worst': st.number_input('Concave Points Worst'),
        'symmetry_worst': st.number_input('Symmetry Worst'),
        'fractal_dimension_worst': st.number_input('Fractal Dimension Worst')
    }   

    # Code for prediction
    diagnosis = ''

    if st.button('Breast Cancer Test Results'):
        diagnosis = breast_cancer_prediction(new_data)
        st.success(diagnosis)

if __name__ == '__main__':
    main()
