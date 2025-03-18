import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import streamlit as st

# Streamlit Title
st.title('Stock Market Prediction for Short Term Investments')

# User Input for Stock Ticker
user_input = st.text_input('Stock Ticker', 'AAPL')

# User Input for Date Range
st.info("Note: Please select a date range within the last 60 days from today.")
date_start = st.text_input('Start Date (YYYY-MM-DD)', '2024-07-03')  
date_end = st.text_input('End Date (YYYY-MM-DD)', '2024-08-31')    

# Fetch stock data using yfinance with error handling
try:
    data = yf.download(user_input, start=date_start, end=date_end, interval='2m')
    
    # Check if the data is empty
    if data.empty:
        st.error(f"Failed to fetch data for {user_input}. The ticker might be incorrect, or the date range might be too narrow.")
        st.stop()
        
except Exception as e:
    st.error(f"Error fetching data for {user_input}: {str(e)}")
    st.stop()

# If data fetched successfully, proceed with displaying it
st.subheader('Data for the last 60 days')
st.dataframe(data.describe(), width=700, height=300)

# Plot Closing Price vs Time
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,7))
plt.plot(data['Close'])
st.pyplot(fig)

# Additional steps for data processing, prediction, etc.
# Continue your existing code below...
