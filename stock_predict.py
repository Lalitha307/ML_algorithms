# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:06:10 2023

@author: DELL
"""

#yesterday is independent feature and today is target feature

############stock price lstm
import pandas as pd
import yfinance as yf
import datetime#time series data and data & time are independent features
from datetime import date, timedelta
today = date.today()#todays date

d1 = today.strftime("%Y-%m-%d")#particular format
end_date = d1#todays date
d2 = date.today() - timedelta(days=5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2#previous date

data = yf.download('AAPL', start=start_date, end=end_date, progress=False)#AAPL is company name

data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
data.tail()

correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))

x = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]
x = x.to_numpy()#converting into numpy array
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)



from keras.models import Sequential
from keras.layers import Dense, LSTM


model = Sequential()#embedding if rnn inplace to sequential lstm
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=30)


#prediction 
import numpy as np
#features = [Open, High, Low, Adj Close, Volume]
features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
model.predict(features)