# app.py
from flask import Flask, render_template
from forecast import run_arima_forecast, run_lstm_forecast

app = Flask(__name__)

@app.route('/')
def index():
    # Run forecasts (this could be done asynchronously or periodically in a real app)
    arima_forecast, arima_mse, arima_image = run_arima_forecast()
    lstm_forecast, lstm_image = run_lstm_forecast()
    
    # For example, extract some key values from forecasts for display
    # You might display today's forecast from ARIMA and LSTM if you wish.
    today_arima = arima_forecast.iloc[-1] if not arima_forecast.empty else "N/A"
    today_lstm = lstm_forecast[-1][0] if lstm_forecast.size > 0 else "N/A"
    
    # Pass these and image paths into the template.
    return render_template('index.html',
                           today_arima=today_arima,
                           today_lstm=today_lstm,
                           arima_image=arima_image,
                           lstm_image=lstm_image)

if __name__ == '__main__':
    app.run(debug=True)
