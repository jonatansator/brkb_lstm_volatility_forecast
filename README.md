# brkb_lstm_volatility_forecast

- This project forecasts the **realized volatility** of **$BRKB** (Berkshire Hathaway) stock using a PyTorch LSTM neural network.
- It includes data preprocessing, LSTM model training, and visualization of actual vs. forecasted volatility.

---

## Files
- `brkb_lstm_volatility_forecast.py`: Main script for training the LSTM and predicting realized volatility.
- `brkb.csv`: Dataset with historical $BRKB stock data.
- `output.png`: Plot.

---

## Libraries Used
- `numpy`
- `pandas`
- `torch` (PyTorch)
- `sklearn` (for MinMaxScaler)
- `plotly`

---

## Timeframe
- **Input**: Uses all available data from `brkb.csv` (assumes historical data up to the latest date in the file).
- **Output**: Forecasts realized volatility for the test period (20% of the dataset, post-training).
