import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Bitcoin Price Prediction", page_icon="", layout="centered")

st.title(" Bitcoin Price Prediction App")
st.write("Enter **today's** market data for Bitcoin to predict **tomorrow's** closing price:")

# Load the trained model safely
model_path = os.path.join(os.path.dirname(__file__), 'btc_model.pkl')

@st.cache_resource
def load_btc_model():
    if os.path.exists('btc_model.pkl'):
        return joblib.load('btc_model.pkl')
    elif os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError("Model file 'btc_model.pkl' not found.")

try:
    model = load_btc_model()
    
    # User inputs simple market data
    st.subheader("Enter Today's Market Data")
    
    col1, col2 = st.columns(2)
    with col1:
        open_price  = st.number_input("Open Price ($)", value=60000.0, step=100.0)
        high_price  = st.number_input("High Price ($)", value=61000.0, step=100.0)
        low_price   = st.number_input("Low Price ($)", value=59000.0, step=100.0)
        
    with col2:
        close_price = st.number_input("Close Price ($)", value=60500.0, step=100.0)
        volume      = st.number_input("Volume ($ / BTC)", value=30000000000.0, step=1000000.0)

    # Automatically calculate technical indicators behind the scenes
    calculated_return = (close_price - open_price) / open_price if open_price != 0 else 0.0
    calculated_ma7    = close_price
    calculated_ma21   = close_price
    calculated_rsi    = 50.0  # Default neutral momentum

    # Prediction Button
    if st.button(" Predict Tomorrow's Price", type="primary"):
        features = np.array([[open_price, high_price, low_price, close_price, volume, 
                              calculated_return, calculated_ma7, calculated_ma21, calculated_rsi]])
        
        prediction = model.predict(features)
        
        st.divider()
        st.success(f" **Predicted Closing Price for Tomorrow:** ${prediction[0]:,.2f}")

except Exception as e:
    st.error(f"Error loading model: {e}")
