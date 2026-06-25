
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ✅ Load the trained model (.keras format)
model = tf.keras.models.load_model("my_model.keras")


# ✅ Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ✅ Custom CSS for background + styling
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4ff, #d9e4f5, #cfe0fc);
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.9);
    border-radius: 10px;
    padding: 10px;
}
h1 {
    color: #0A1F44;
    text-shadow: 1px 1px 3px #b0c4de;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# ✅ Streamlit App
st.title('🏦 Customer Churn Prediction Dashboard')
st.markdown("✨ Enter customer details below to check churn probability.")

# User input
geography = st.selectbox('🌍 Select Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Select Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92)
balance = st.number_input('💰 Account Balance')
credit_score = st.number_input('💳 Credit Score')
estimated_salary = st.number_input('💵 Estimated Salary')
tenure = st.slider('📅 Tenure (Years with Bank)', 0, 10)
num_of_products = st.slider('📦 Number of Products', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.subheader(f'🔮 Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.error('⚠️ The customer is **likely to churn**.')
else:
    st.success('✅ The customer is **not likely to churn**.')
