import streamlit as st
import pandas as pd
import joblib

# Set page layout for Mobile Friendly view
st.set_page_config(page_title="Bitcoin Price Predictor", page_icon="", layout="centered")

st.title(" Bitcoin Close Price Predictor")
st.write("Ingiza data za soko la leo ili kutabiri bei ya kufunga (Close Price):")

# 1. Pakia Model
model = joblib.load('btc_model.pkl')

# 2. Form ya Kuingiza Data kwenye Simu
with st.form("prediction_form"):
    open_price = st.number_input("Open Price ($ USD)", value=63853.49, step=100.0)
    high_price = st.number_input("High Price ($ USD)", value=64620.41, step=100.0)
    low_price = st.number_input("Low Price ($ USD)", value=62714.80, step=100.0)
    volume = st.number_input("Volume ($ USD)", value=24600000000.0, step=1000000.0)
    
    submit_btn = st.form_submit_button(" Tabiri Bei (Predict)")

# 3. Onyesha Matokeo
if submit_btn:
    input_data = pd.DataFrame([{
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Volume': volume
    }])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"###  Estimated Close Price: ${float(prediction.flat[0]):,.2f} USD")
