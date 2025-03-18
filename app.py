import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import math
from sklearn.metrics import mean_squared_error
import yfinance as yf
from keras.models import load_model
import streamlit as st
import os

st.title('Stock Market Prediction for Short Term Investments')

# User input for stock ticker and date range
user_input = st.text_input('Stock Ticker', 'AAPL')
st.info("Note: Please select a date range within the last 60 days from today.")
st.subheader('Data for the last 60 days')

# Input fields for date range
date_start = st.text_input('Start Date (YYYY-MM-DD)', '2024-07-03')  
date_end = st.text_input('End Date (YYYY-MM-DD)', '2024-08-31')

# Date validation
try:
    pd.to_datetime(date_start)
    pd.to_datetime(date_end)
except ValueError:
    st.error("Invalid date format. Please enter dates in YYYY-MM-DD format.")
    st.stop()

# Download stock data
data = yf.download(user_input, start=date_start, end=date_end, interval='2m')

# Error handling if data is not fetched
if data.empty:
    st.error(f"Failed to fetch data for {user_input}. Check ticker symbol or adjust the date range.")
    st.stop()

# Display summary statistics
st.dataframe(data.describe(), width=700, height=300)

# Plotting closing price
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 7))
plt.plot(data['Close'])
st.pyplot(fig)

# Data preprocessing for LSTM
df = data.reset_index()['Close']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(df).reshape(-1, 1))

# Splitting the dataset into training and testing sets
train_size = int(len(df) * 0.7)
test_size = len(df) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(df), :1]

# Function to create dataset for LSTM
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:i + time_step, 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# Reshape data for LSTM (3D shape required)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Model loading check
if not os.path.exists('StockModel.h5'):
    st.error("Model file 'StockModel.h5' not found!")
    st.stop()

# Load pre-trained model
model = load_model('StockModel.h5')

# Predicting with the trained model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
Y_test = scaler.inverse_transform(test_data.reshape(-1, 1))

# Plotting Actual vs Predicted (Training Data)
st.subheader('Training Data: Actual vs Predicted')
plt.figure(figsize=(12, 5))
plt.plot(Y_train, label='Actual Training Data', color='blue')
plt.plot(train_predict, label='Predicted Training Data', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
st.pyplot(plt)

# Plotting Actual vs Predicted (Testing Data)
st.subheader('Testing Data: Actual vs Predicted')
plt.figure(figsize=(12, 5))
plt.plot(Y_test, label='Actual Testing Data', color='green')
plt.plot(test_predict, label='Predicted Testing Data', color='yellow')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
st.pyplot(plt)

# For next 10 days prediction
fornext10days = 1072

# Create prediction column
data['Prediction'] = data[['Close']].shift(-fornext10days)
data.tail()

# Prepare data for next 10 days prediction
actual_size = len(data) - fornext10days
pred_size = fornext10days
actual_data = data['Prediction'][:actual_size].values.reshape(-1, 1)
pred_data = data['Prediction'][actual_size:].values.reshape(-1, 1)

scaled_actualdata = scaler.fit_transform(np.array(actual_data).reshape(-1, 1))

def create_dataset_10(dataset, time_step):
    dataX, dataY = [], []
    for i in range(time_step, len(dataset)):
        dataX.append(dataset[i - time_step:i, 0])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
actual_x, actual_y = create_dataset_10(scaled_actualdata, time_step)
actual_x = actual_x.reshape(actual_x.shape[0], actual_x.shape[1], 1)

# Initialize testing data with the last 'time_step' values
testing_data = scaled_actualdata[-time_step:].reshape(1, time_step, 1)
future_predictions = []

# Predict future prices
for _ in range(fornext10days):
    pred_value = model.predict(testing_data)
    pred_value_inversed = scaler.inverse_transform(pred_value)
    future_predictions.append(pred_value_inversed[0, 0])
    
    # Update testing data with the new prediction
    new_input = np.append(testing_data[0, 1:], pred_value)
    testing_data = new_input.reshape(1, time_step, 1)

future_predictions = np.array(future_predictions)

# Save future predictions model (optional)
model.save('next10days.h5')

# Display predicted plot for the next 10 days
st.subheader('Predicted Stock Prices for the Next 10 Days (Disclaimer: This is for study purposes, invest at your own risk)')

# Plotting future predictions
plt.figure(figsize=(12, 7))
plt.plot(future_predictions, label='Predicted Stock Price', color='green')
plt.title('Predicted Stock Prices for the Next 10 Days')
plt.xlabel('Time')
plt.ylabel('Predicted Close Price')
plt.legend()
st.pyplot(plt)
