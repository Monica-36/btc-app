import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="BTC Price Prediction", page_icon="🚀")

st.title(" Bitcoin Price Prediction App")
st.write("Weka takwimu za leo za Bitcoin ili kutabiri bei ya Close ya kesho:")

# Safisha path ya model
model_path = os.path.join(os.path.dirname(__file__), 'btc_model.pkl')

@st.cache_resource
def load_btc_model():
    if os.path.exists('btc_model.pkl'):
        return joblib.load('btc_model.pkl')
    elif os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError("Faili la 'btc_model.pkl' halikupatikana kwenye repository.")

try:
    model = load_btc_model()
    
    st.subheader("1. Takwimu za Mwanzo (Prices & Volume)")
    col1, col2 = st.columns(2)
    
    with col1:
        open_price = st.number_input("Open Price ($)", value=60000.0, step=100.0)
        high_price = st.number_input("High Price ($)", value=61000.0, step=100.0)
        low_price  = st.number_input("Low Price ($)", value=59000.0, step=100.0)
        
    with col2:
        close_price = st.number_input("Close Price ($)", value=60500.0, step=100.0)
        volume      = st.number_input("Volume", value=30000000000.0, step=1000000.0)

    st.subheader("2. Indicators za Kiufundi (Technical Indicators)")
    col3, col4 = st.columns(2)
    
    with col3:
        ret = st.number_input("Return (pct_change)", value=0.01, format="%.4f")
        ma7 = st.number_input("7-Day Moving Average (MA7)", value=60000.0, step=100.0)
        
    with col4:
        ma21 = st.number_input("21-Day Moving Average (MA21)", value=59500.0, step=100.0)
        rsi  = st.number_input("RSI (14)", value=55.0, min_value=0.0, max_value=100.0)

    # Prediction Button
    if st.button("🔮 Tabiri Bei ya Kesho", type="primary"):
        features = np.array([[open_price, high_price, low_price, close_price, volume, ret, ma7, ma21, rsi]])
        prediction = model.predict(features)
        
        st.success(f" **Utabiri wa Bei ya Close ya Kesho:** ${prediction[0]:,.2f}")

except Exception as e:
    st.error(f"Kuna tatizo wakati wa kupakia model: {e}")
