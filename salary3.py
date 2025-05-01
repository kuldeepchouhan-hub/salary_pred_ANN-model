import tensorflow as tf
import streamlit as st 
import pandas as pd 
import pickle 
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load model and files with 
model = tf.keras.models.load_model('model.h5')
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler_X.pkl', 'rb') as file:
    scaler_x = pickle.load(file)
with open('scaler_y.pkl', 'rb') as file:
    scaler_y = pickle.load(file)

# Streamlit
st.title("Fintech Customer Bank Salary Prediction")

# User inputs 
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0, step=250898.08)
credit_score = st.number_input("Credit Score", min_value=350.0, max_value=850.0, step=1.0)
tenure = st.number_input('Tenure', 0, 10)
num_of_products = st.number_input('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine encoded and numerical input
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale and predict
input_data_scaled = scaler_x.transform(input_data)
prediction = model.predict(input_data_scaled)
predicted_salary = scaler_y.inverse_transform(prediction)[0][0]

# Display result
st.success(f"Predicted Estimated Salary: ${predicted_salary:,.2f}")




