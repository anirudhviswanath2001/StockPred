import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
import math
from sklearn.metrics import mean_squared_error
import yfinance as yf
from keras.models import load_model
import streamlit as st



st.title('Stock Market Prediction for Short Term Investments')
user_input=st.text_input('Stock Ticker','AAPL')
date_start=st.text_input('YYYY-MM-DD','2024-07-01')
date_end=st.text_input('YYYY-MM-DD','2024-08-29')
data=yf.download(user_input,date_start,date_end,interval='2m')

#Describing the data
st.subheader('Data for the last 60 days')
# Display summary statistics with custom size
st.dataframe(data.describe(), width=700, height=300)
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,7))
plt.plot(data.Close)
st.pyplot(fig)

df=data.reset_index()['Close']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(df).reshape(-1,1))

## splitting the dataset into twor prts training and testing data
train_size=int(len(df)*0.7)
test_size=len(df)-train_size
train_data,test_data=scaled_data[0:train_size,:],scaled_data[train_size:len(df),:1]


def create_dataset(dataset,time_step):
    dataX, dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:i+(time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX),np.array(dataY)

time_step=100
X_train, Y_train=create_dataset(train_data,time_step)
X_test, Y_test=create_dataset(test_data,time_step)

#reshape input to be LSTM which is 3Dimensional
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1) 

@st.cache_resource
def load_model():
    return tensorflow.keras.models.load_model('StockModel.h5')

model = load_model()


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
Y_test = scaler.inverse_transform(test_data.reshape(-1, 1))

st.subheader('Training Data: Actual vs Predicted')
plt.figure(figsize=(12, 5))
plt.plot(Y_train, label='Actual Training Data', color='blue')
plt.plot(train_predict, label='Predicted Training Data', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
# Display the plot using Streamlit's plotting function
st.pyplot(plt)

# Plot for Testing Data: Actual vs Predicted
st.subheader('Testing Data: Actual vs Predicted')
plt.figure(figsize=(12, 5))
plt.plot(Y_test, label='Actual Testing Data', color='green')
plt.plot(test_predict, label='Predicted Testing Data', color='yellow')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
# Display the plot using Streamlit's plotting function
st.pyplot(plt)

#for next 10 days prediction
fornext10days=1072
#creating a column shifted to 'x' units/days   
data['Prediction']=data[['Close']].shift(-fornext10days)
data.tail() 
actual_size=int(len(data)-fornext10days)
pred_size=int(fornext10days)
actual_data=data['Prediction'][:actual_size].values.reshape(-1,1)
pred_data=data['Prediction'][actual_size:].values.reshape(-1,1),
scaled_actualdata = scaler.fit_transform(np.array(actual_data).reshape(-1,1))
def create_dataset_10(dataset,time_step):
    dataX, dataY = [], []
    for i in range(time_step, len(dataset)):
        dataX.append(dataset[i - time_step:i, 0])
        dataY.append(dataset[i, 0])
    return np.array(dataX),np.array(dataY)
time_step=100
actual_x,actual_y=create_dataset_10(scaled_actualdata,time_step)
actual_x=actual_x.reshape(actual_x.shape[0],actual_x.shape[1],1)
# Initialize testing_data with the last 'time_step' values from the scaled dataset, reshaped to match the model input shape
testing_data = scaled_actualdata[-time_step:].reshape(1, time_step, 1)
future_predictions = []
for _ in range(1072):  # Number of future days to predict
    # Make a prediction
    pred_value = model.predict(testing_data)
    
    # Inverse scale the prediction back to original value
    pred_value_inversed = scaler.inverse_transform(pred_value)
    
    # Store the prediction
    future_predictions.append(pred_value_inversed[0, 0])
    
    # Update testing_data by appending the new prediction and dropping the first value
    new_input = np.append(testing_data[0, 1:], pred_value)
    testing_data = new_input.reshape(1, time_step, 1)
future_predictions = np.array(future_predictions)
model.save('next10days.h5')

# Set a subheader for the predicted plot
st.subheader('Predicted Stock Prices for the Next 10 Days(Disclaimer: This is only for study purpose invest in stocks at your own risk)')

# Create a figure for the predictions
plt.figure(figsize=(12, 7))
plt.plot(future_predictions, label='Predicted Stock Price', color='green')
plt.title('Predicted Stock Prices for the Next 10 Days')
plt.xlabel('Time')
plt.ylabel('Predicted Close Price')
plt.legend()

# Display the plot using Streamlit's plotting function
st.pyplot(plt)

