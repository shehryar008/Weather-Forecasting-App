import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime
import matplotlib
matplotlib.use('Agg')


def run_arima_forecast():
    # Load CSV file (update path if needed)
    data = pd.read_csv('weather.csv')
    # (Because your CSV has no date, create a synthetic index with the most recent date as today)
    data.rename(columns={'MaxTemp': 'Temperature'}, inplace=True)
    num_rows = len(data)
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=num_rows-1)
    date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    data.index = date_index
    data = data.asfreq('D')
    data['Temperature'].fillna(method='ffill', inplace=True)
    
    # Split data into train/test
    temp_series = data['Temperature']
    train_size = int(len(temp_series) * 0.8)
    train, test = temp_series[:train_size], temp_series[train_size:]
    
    # Fit an ARIMA model – adjust parameters as needed
    model_arima = ARIMA(train, order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    forecast_arima = model_arima_fit.forecast(steps=len(test))
    
    mse_arima = mean_squared_error(test, forecast_arima)
    # You could also generate a plot, save it as an image in /static/images, then pass the file path.
    plt.figure(figsize=(10,5))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast_arima, label='Forecast', color='red')
    plt.title(f'ARIMA Forecast\nMSE: {mse_arima:.3f}')
    plt.legend()
    image_path = r'static\arima-forecast-visualization.svg'
    plt.savefig(image_path)
    plt.close()
    return forecast_arima, mse_arima, image_path

def run_lstm_forecast():
    # Similar process for LSTM forecast
    data = pd.read_csv('weather.csv')
    data.rename(columns={'MaxTemp': 'Temperature'}, inplace=True)
    num_rows = len(data)
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=num_rows-1)
    date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    data.index = date_index
    data = data.asfreq('D')
    data['Temperature'].fillna(method='ffill', inplace=True)

    temperatures = data['Temperature'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    temps_scaled = scaler.fit_transform(temperatures)

    # Create the dataset for LSTM
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 3
    X, Y = create_dataset(temps_scaled, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, input_shape=(look_back, 1)))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    
    model_lstm.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose=0)
    predictions_lstm = model_lstm.predict(X_test)
    predictions_lstm = predictions_lstm.reshape(-1, 1)
    predictions_actual = scaler.inverse_transform(predictions_lstm)
    Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
    
    # Generate and save a plot
    plt.figure(figsize=(10,5))
    plt.plot(Y_test_actual, label='Actual')
    plt.plot(predictions_actual, label='Forecast', color='red')
    plt.title('LSTM Forecast')
    plt.legend()
    image_path = 'static\lstm-forecast-visualization.svg'
    plt.savefig(image_path)
    plt.close()
    return predictions_actual, image_path
