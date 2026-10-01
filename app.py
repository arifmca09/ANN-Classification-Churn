import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# --- LOAD MODEL & ARTIFACTS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.keras')
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        label_encoder_geo = pickle.load(file)

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        
    return model, label_encoder_geo, label_encoder_gender, scaler

model, label_encoder_geo, label_encoder_gender, scaler = load_assets()

# --- STREAMLIT UI ---
st.title('Customer Churn Prediction')

# User inputs
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# --- PREDICTION LOGIC ---
if st.button('Predict Churn', type='primary'):
    # Prepare the input data (include Geography temporarily so we can encode it)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-Hot encode 'Geography' safely
    geo_encoded = label_encoder_geo.transform([[input_data['Geography'].iloc[0]]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

    # Drop old 'Geography' text column and combine with one-hot columns
    input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict Churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])

    # Display results on the web app screen
    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prediction_proba:.2%}")
    
    if prediction_proba > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")