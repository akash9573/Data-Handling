# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Step 1: Load Data
# Fetch historical data for Apple Inc. (AAPL)
data = yf.download('AAPL', start='2010-01-01', end='2013-01-01')

# Step 2: Data Cleaning
# Remove any missing values
data = data.dropna()

# Step 3: Data Processing
# Calculate moving averages
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Step 4: EDA Analysis
# Plot closing prices and moving averages
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['SMA_50'], label='50-Day SMA')
plt.plot(data['SMA_200'], label='200-Day SMA')
plt.title('AAPL Closing Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot volume traded
plt.figure(figsize=(14, 7))
plt.plot(data['Volume'], label='Volume Traded', color='orange')
plt.title('AAPL Volume Traded Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()

# Step 5: Feature Engineering
# Add lag features (previous day prices)
data['Lag_1'] = data['Close'].shift(1)
data['Lag_2'] = data['Close'].shift(2)

# Drop rows with missing values due to lag features
data = data.dropna()

# Step 6: Preparing Data for Prediction
# Define the feature set and target variable
features = ['Lag_1', 'Lag_2']
target = 'Close'

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Building the Prediction Model
# Using Linear Regression for prediction
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate the Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f'Linear Regression MAE: {mae_lr}')
print(f'Linear Regression RMSE: {rmse_lr}')

# Step 8: Building an LSTM Model
# Reshape data for LSTM model
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, X_train_lstm.shape[2])))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X_train_lstm, y_train, batch_size=1, epochs=1)

# Make predictions
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = y_pred_lstm.flatten()

# Evaluate the LSTM model
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))

print(f'LSTM Model MAE: {mae_lstm}')
print(f'LSTM Model RMSE: {rmse_lstm}')

# Step 9: Visualization of Predictions
plt.figure(figsize=(14, 7))
plt.plot(data.index[len(data) - len(y_test):], y_test, color='blue', label='Actual Prices')
plt.plot(data.index[len(data) - len(y_test):], y_pred_lr, color='red', label='Linear Regression Predictions')
plt.plot(data.index[len(data) - len(y_test):], y_pred_lstm, color='green', label='LSTM Predictions')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
