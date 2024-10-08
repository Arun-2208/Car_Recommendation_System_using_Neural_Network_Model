import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained Keras model
model = load_model('final_car_recommendation_model_optimized.keras')

# Define the feature columns that were used for training the model
feature_columns = [
    'Price ($)', 'Year of Manufacture', 'Resale Value ($)', 'Fuel Economy (km/l)', 
    'Performance Rating', 'User Rating', 'Safety Rating', 'Comfort Level', 
    'Maintenance Cost ($/yr)', 'Warranty Period (years)', 'Seating Capacity', 
    'Battery Capacity (kWh)', 'Body Type_Convertible', 'Body Type_Coupe', 'Body Type_Hatchback',
    'Body Type_Minivan', 'Body Type_SUV', 'Body Type_Sedan', 'Body Type_Wagon',
    'Engine Type_Diesel', 'Engine Type_Electric', 'Engine Type_Hybrid', 
    'Engine Type_Petrol', 'Transmission Type_Automatic', 'Transmission Type_Manual', 
    'Drive Type_AWD', 'Drive Type_FWD', 'Drive Type_RWD'
]

# Sample options for user selections
engine_options = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
body_type_options = ['Convertible', 'Coupe', 'Hatchback', 'Minivan', 'SUV', 'Sedan', 'Wagon']
transmission_options = ['Automatic', 'Manual']
drive_type_options = ['AWD', 'FWD', 'RWD']

# Function to process user input (similar to how data was prepared)
def process_user_input(user_input):
    df_user_input = pd.DataFrame([user_input])

    # Define the numerical and categorical columns
    numerical_columns = ['Price ($)', 'Year of Manufacture', 'Resale Value ($)', 'Fuel Economy (km/l)', 
                         'Performance Rating', 'User Rating', 'Safety Rating', 'Comfort Level', 
                         'Maintenance Cost ($/yr)', 'Warranty Period (years)', 'Seating Capacity', 
                         'Battery Capacity (kWh)']
    
    categorical_columns = ['Body Type', 'Engine Type', 'Transmission Type', 'Drive Type']

    # Normalize numerical values using MinMaxScaler (same as training process)
    scaler = MinMaxScaler()
    df_numerical = df_user_input[numerical_columns].astype(float)
    df_numerical_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=numerical_columns)

    # Handle categorical columns using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    df_categorical_encoded = pd.DataFrame(encoder.fit_transform(df_user_input[categorical_columns]), 
                                          columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the scaled numerical data and the one-hot encoded categorical data
    df_final_input = pd.concat([df_numerical_scaled, df_categorical_encoded], axis=1)

    # Ensure the processed input has 28 features
    if df_final_input.shape[1] != 28:
        st.error("Processed input does not have the required 28 features.")
        return None

    return df_final_input

# Streamlit app layout and user input collection
st.set_page_config(page_title="Car Purchase Recommendation System", layout="centered")
st.markdown('<div class="main-title">Car Purchase Recommendation System</div>', unsafe_allow_html=True)

# Function to get user input
def get_user_input():
    user_input = {
        'Year of Manufacture': st.slider('Year of Manufacture', 2010, 2024, 2020),
        'Price ($)': st.slider('Price ($)', 10000, 200000, 50000, step=5000),
        'Resale Value ($)': st.slider('Resale Value ($)', 5000, 50000, 20000, step=1000),
        'Fuel Economy (km/l)': st.slider('Fuel Economy (km/l)', 8, 30, 15),
        'Performance Rating': st.slider('Performance Rating', 1, 10, 5),
        'User Rating': st.slider('User Rating', 1, 10, 5),
        'Safety Rating': st.slider('Safety Rating', 1, 10, 5),
        'Comfort Level': st.slider('Comfort Level', 1, 10, 5),
        'Maintenance Cost ($/yr)': st.slider('Maintenance Cost ($/yr)', 500, 7000, 1500, step=500),
        'Warranty Period (years)': st.slider('Warranty Period (years)', 1, 5, 3),
        'Seating Capacity': st.selectbox('Seating Capacity', [4, 5, 7]),
        'Battery Capacity (kWh)': st.slider('Battery Capacity (kWh)', 20, 100, 50),
        'Body Type': st.selectbox('Body Type', body_type_options),
        'Engine Type': st.selectbox('Engine Type', engine_options),
        'Transmission Type': st.selectbox('Transmission Type', transmission_options),
        'Drive Type': st.selectbox('Drive Type', drive_type_options)
    }

    return user_input

# Main execution flow
user_input = get_user_input()

if st.button("Submit"):
    st.write("Your preferences have been recorded. Calculating the best car recommendation for you...")
    
    processed_input = process_user_input(user_input)
    
    # Feed the processed user input into the model for prediction
    prediction = model.predict(processed_input)
    recommended_car_index = np.argmax(prediction, axis=1)
    
    # Map predicted index to car name
    car_names = df_final['Car'].unique()  # Assuming these are in the dataset
    recommended_car = car_names[recommended_car_index[0]]
    
    st.markdown(f"### Recommended Car: {recommended_car}")

