import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf

st.title("Stock Price Predictor")
stock = st.text_input("Enter the Stock ID", "GOOG")
from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)
google_data = yf.download(stock, start, end)
model = load_model("Latest_stockprice_model.keras")
st.subheader("Stock Data")
st.write(google_data)
splittinglen=int(len(google_data)*0.7)
xtest=pd.DataFrame(google_data.Close[splittinglen:])

def plotgraph(figsize, values, fulldata, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(fulldata.Close,'b')
    if extra_dataset:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plotgraph((15,6),google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plotgraph((15,6),google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plotgraph((15,6),google_data['MA_for_100_days'],google_data,0))

# st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
# st.pyplot(plotgraph((15,6),google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaleddata= scaler.fit_transform(xtest[['Close']])

xdata = []
ydata = []

for i in range(100, len(scaleddata)):
    xdata.append(scaleddata[i-100:i])
    ydata.append(scaleddata[i])
    
xdata,ydata=np.array(xdata), np.array(ydata)

predictions = model.predict(xdata)
invpredictions = scaler.inverse_transform(predictions)
inv_ytest = scaler.inverse_transform(ydata)

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_ytest.reshape(-1),
        'predictions': invpredictions.reshape(-1)
    }, index = google_data.index[splittinglen+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader("Original Close price vs Predicted Close price")
fig = plt.figure(figsize=(8,6))
plt.plot(pd.concat([google_data.Close[:splittinglen+100],ploting_data],axis=0))
plt.legend(["Data- not used", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)