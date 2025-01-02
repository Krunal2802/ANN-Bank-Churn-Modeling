import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

## load trained model
model = load_model('regressionModel.h5')

## load encoder
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

## load scaler
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file) 

# to run streamlit app: 
# command: streamlit run app.py

## streamlit WebApp
st.title('Estimated Salary Prediction')

## user input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
# exited = st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    # 'Exited': [exited],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

st.write("Input data columns:", input_data.columns)
st.write("Expected columns:", scaler.feature_names_in_)

# Add missing columns if necessary
missing_columns = set(scaler.feature_names_in_) - set(input_data.columns)
for col in missing_columns:
    input_data[col] = 0

# Reorder columns to match training data
input_data = input_data[scaler.feature_names_in_]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict salary
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Predicted Estimaetd Salary: ${prediction_proba:.2f}')