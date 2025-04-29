# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration dictionary for flexibility across markets
CONFIG = {
    'markets': ['EURUSD=X', 'AAPL', '^GSPC'],  # Forex, Stock, Index
    'start_date': '2014-01-01',
    'end_date': '2024-01-01',
    'short_window': 50,
    'long_window': 200,
    'lookback': 20,
    'entry_threshold': 1.5,
    'exit_threshold': 0.5,
    'max_position': 1,
    'stop_loss': 0.02,
    'autoencoder_epochs': 50,
    'autoencoder_batch_size': 32,
    'encoding_dim': 3,
    'model_epochs': 20,
    'model_batch_size': 32
}

# Step 1: Data Collection
def get_market_data(tickers, start_date, end_date):
    """
    Fetches historical data for multiple markets using yfinance.
    Returns a dictionary of DataFrames.
    """
    data_dict = {}
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, ignore_tz=True)
            if data.empty:
                print(f"Warning: No data retrieved for {ticker}.")
                continue

            # Select and rename columns
            pair_data = data[['Close', 'Volume']].rename(columns={'Close': 'Price'}).copy()
            pair_data.index = pd.to_datetime(pair_data.index)

            # Ensure numeric data and handle NaNs
            pair_data['Volume'] = pd.to_numeric(pair_data['Volume'], errors='coerce').fillna(0)
            pair_data['Price'] = pd.to_numeric(pair_data['Price'], errors='coerce')
            pair_data = pair_data.dropna(subset=['Price'])

            if not pair_data.empty:
                print(f"-> Successfully processed {len(pair_data)} data points for {ticker}")
                data_dict[ticker] = pair_data
            else:
                print(f"-> No valid data after processing for {ticker}.")

        except Exception as e:
            print(f"Error during data acquisition for {ticker}: {e}")
            continue

    return data_dict

# Step 2: Data Preprocessing
def preprocess_data(df):
    """
    Preprocesses the data by handling missing values, ensuring proper data types, and sorting by date.
    """
    try:
        # Sort by date
        df = df.sort_index()

        # Forward fill any remaining NaNs in Price
        df['Price'] = df['Price'].ffill()

        # Ensure Volume is zero where NaN
        df['Volume'] = df['Volume'].fillna(0)

        print(f"Data after preprocessing: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return pd.DataFrame()

# Step 3: Moving Average Spread Creation
def create_moving_average_spread(df, short_window, long_window):
    """
    Creates synthetic spreads using moving averages.
    Adds MA, synthetic spread, MA spread, and Z-scores.
    """
    try:
        df['MA_Short'] = df['Price'].rolling(window=short_window).mean()
        df['MA_Long'] = df['Price'].rolling(window=long_window).mean()

        # Synthetic Spread: Price - MA_Short
        df['Synthetic_Spread'] = df['Price'] - df['MA_Short']

        # Moving Average Spread: MA_Short - MA_Long
        df['MA_Spread'] = df['MA_Short'] - df['MA_Long']

        # Z-Score of Synthetic Spread
        df['Spread_ZScore'] = (df['Synthetic_Spread'] - df['Synthetic_Spread'].rolling(window=short_window).mean()) / df['Synthetic_Spread'].rolling(window=short_window).std()

        # Drop rows with NaN values due to rolling calculations
        df = df.dropna()
        print(f"Data after spread creation: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error during spread creation: {e}")
        return pd.DataFrame()

# Step 4: Feature Engineering
def feature_engineering(df):
    """
    Engineers features including log returns, technical indicators, moving average crosses, and volatility.
    """
    try:
        # Log Returns
        df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))

        # Technical Indicators using ta-lib
        df['RSI'] = ta.momentum.RSIIndicator(df['Price'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df['Price']).macd()
        df['Bollinger_Upper'], df['Bollinger_Lower'] = ta.volatility.BollingerBands(df['Price']).bollinger_hband(), ta.volatility.BollingerBands(df['Price']).bollinger_lband()

        # Moving Average Cross: 1 if MA_Short > MA_Long, else 0
        df['MA_Cross'] = (df['MA_Short'] > df['MA_Long']).astype(int)

        # Volatility (Rolling Standard Deviation of Log Returns)
        df['Volatility'] = df['Log_Returns'].rolling(window=20).std()

        # Drop rows with NaN values
        df = df.dropna()
        print(f"Data after feature engineering: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return pd.DataFrame()

# Step 5: Autoencoder Training
def train_autoencoder(df, features, encoding_dim, epochs, batch_size):
    """
    Trains an Autoencoder for dimensionality reduction and anomaly detection.
    Returns the encoded features and reconstruction errors.
    """
    try:
        # Select features for Autoencoder
        X = df[features].values

        # Scale the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Define Autoencoder architecture
        input_dim = len(features)
        autoencoder = Sequential([
            Input(shape=(input_dim,)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(encoding_dim, activation='relu'),  # Encoded layer
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(input_dim, activation='sigmoid')
        ])

        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0)

        # Get encoded features
        encoder = Sequential(autoencoder.layers[:3])
        encoded_features = encoder.predict(X_scaled, verbose=0)

        # Get reconstruction errors for anomaly detection
        reconstructed = autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)
        df['Reconstruction_Error'] = mse

        # Add encoded features to DataFrame
        for i in range(encoding_dim):
            df[f'Encoded_Feature_{i}'] = encoded_features[:, i]

        print(f"Autoencoder trained. Added {encoding_dim} encoded features and reconstruction errors.")
        return df, scaler
    except Exception as e:
        print(f"Error during Autoencoder training: {e}")
        return df, None

# Step 6: Prepare Data for Models
def prepare_data_for_models(df, features, target='Spread_ZScore', lookback=20):
    """
    Prepares data for time series modeling by creating sequences for LSTM/CNN-LSTM.
    Returns scaled features and target for all models.
    """
    try:
        # Select features and target
        X = df[features].values
        y = df[target].values

        # Scale features and target
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Create sequences for LSTM/CNN-LSTM
        X_seq, y_seq = [], []
        for i in range(lookback, len(X_scaled)):
            X_seq.append(X_scaled[i-lookback:i])
            y_seq.append(y_scaled[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # Split data (80% train, 20% test)
        train_size = int(len(X_seq) * 0.8)
        X_train_seq, X_test_seq = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

        # For traditional ML models, use flattened data
        X_flat = X_scaled[lookback:]
        X_train_flat, X_test_flat = X_flat[:train_size], X_flat[train_size:]

        return (X_train_seq, X_test_seq, X_train_flat, X_test_flat, y_train, y_test, feature_scaler, target_scaler)
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return None, None, None, None, None, None, None, None

# Step 7: Train Models
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Flatten(),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_models(X_train_seq, X_train_flat, y_train, input_shape, epochs, batch_size):
    """
    Trains all models (LSTM, CNN-LSTM, XGBoost, Random Forest, OLS).
    Returns trained models.
    """
    try:
        # LSTM
        lstm_model = build_lstm_model(input_shape)
        lstm_model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # CNN-LSTM
        cnn_lstm_model = build_cnn_lstm_model(input_shape)
        cnn_lstm_model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_flat, y_train)

        # XGBoost
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train_flat, y_train)

        # OLS
        X_train_flat_ols = sm.add_constant(X_train_flat)
        ols_model = sm.OLS(y_train, X_train_flat_ols).fit()

        return {
            "LSTM": lstm_model,
            "CNN-LSTM": cnn_lstm_model,
            "RandomForest": rf_model,
            "XGBoost": xgb_model,
            "OLS": ols_model
        }
    except Exception as e:
        print(f"Error during model training: {e}")
        return {}

# Step 8: Signal Generation
def generate_signals(df, predictions, target_scaler, z_score_col='Spread_ZScore', entry_threshold=1.5, exit_threshold=0.5):
    """
    Generates long/short trading signals based on predicted spread Z-scores.
    """
    try:
        # Inverse transform predictions to original scale
        predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Align predictions with the original DataFrame
        signal_df = df.iloc[-len(predictions):].copy()
        signal_df['Predicted_ZScore'] = predictions

        # Initialize signals
        signal_df['Signal'] = 0
        signal_df['Position'] = 0
        position = 0

        for i in range(1, len(signal_df)):
            z_score = signal_df['Predicted_ZScore'].iloc[i]
            if z_score > entry_threshold and position == 0:
                signal_df['Signal'].iloc[i] = -1  # Short
                position = -1
            elif z_score < -entry_threshold and position == 0:
                signal_df['Signal'].iloc[i] = 1   # Long
                position = 1
            elif abs(z_score) < exit_threshold and position != 0:
                signal_df['Signal'].iloc[i] = 0   # Exit
                position = 0
            signal_df['Position'].iloc[i] = position

        return signal_df
    except Exception as e:
        print(f"Error during signal generation: {e}")
        return pd.DataFrame()

# Step 9: Walk-Forward Backtest
def backtest_strategy(signals_df, price_col='Price'):
    """
    Performs a walk-forward backtest to evaluate the trading strategy.
    Returns cumulative returns and other metrics.
    """
    try:
        df = signals_df.copy()
        df['Returns'] = df[price_col].pct_change()
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod() - 1
        return df
    except Exception as e:
        print(f"Error during backtesting: {e}")
        return pd.DataFrame()

# Step 10: Risk Management
def apply_risk_management(df, max_position, stop_loss):
    """
    Applies risk management rules such as position limits and stop-loss.
    """
    try:
        df['Position'] = df['Position'].clip(-max_position, max_position)
        df['Stop_Loss'] = 0
        position = 0
        entry_price = None

        for i in range(1, len(df)):
            if df['Position'].iloc[i-1] != 0 and position == 0:
                position = df['Position'].iloc[i-1]
                entry_price = df['Price'].iloc[i]
            elif position != 0:
                price_change = (df['Price'].iloc[i] - entry_price) / entry_price
                if (position == 1 and price_change <= -stop_loss) or (position == -1 and price_change >= stop_loss):
                    df['Stop_Loss'].iloc[i] = 1
                    position = 0
                    entry_price = None
            df['Position'].iloc[i] = position if position != 0 else df['Position'].iloc[i]

        # Recalculate returns after risk management
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod() - 1
        return df
    except Exception as e:
        print(f"Error during risk management: {e}")
        return pd.DataFrame()

# Step 11: Performance Evaluation
def evaluate_performance(df):
    """
    Evaluates the performance of the trading strategy using various metrics.
    """
    try:
        returns = df['Strategy_Returns'].dropna()
        if len(returns) == 0:
            return {"Sharpe_Ratio": 0, "Cumulative_Return": 0, "Max_Drawdown": 0}

        # Sharpe Ratio (annualized, assuming 252 trading days)
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))

        # Cumulative Return
        cumulative_return = df['Cumulative_Returns'].iloc[-1]

        # Maximum Drawdown
        rolling_max = (1 + df['Cumulative_Returns']).cummax()
        drawdown = (1 + df['Cumulative_Returns']) / rolling_max - 1
        max_drawdown = drawdown.min()

        return {
            "Sharpe_Ratio": sharpe_ratio,
            "Cumulative_Return": cumulative_return,
            "Max_Drawdown": max_drawdown
        }
    except Exception as e:
        print(f"Error during performance evaluation: {e}")
        return {"Sharpe_Ratio": 0, "Cumulative_Return": 0, "Max_Drawdown": 0}

# Step 12: Main Pipeline
def run_pipeline(config):
    """
    Runs the entire pipeline for all markets and compares performance.
    """
    # Fetch data for all markets
    data_dict = get_market_data(config['markets'], config['start_date'], config['end_date'])
    if not data_dict:
        print("No data available for any market. Exiting.")
        return

    # Dictionary to store results for each market
    all_results = {}
    all_metrics = {}

    # Define features for models
    autoencoder_features = ['Log_Returns', 'RSI', 'MACD', 'Volatility', 'Spread_ZScore']
    model_features = ['Log_Returns', 'RSI', 'MACD', 'Volatility', 'Spread_ZScore', 'MA_Cross', 'Reconstruction_Error'] + [f'Encoded_Feature_{i}' for i in range(config['encoding_dim'])]

    for ticker, data in data_dict.items():
        print(f"\nProcessing market: {ticker}")
        # Preprocess data
        data = preprocess_data(data)
        if data.empty:
            print(f"Skipping {ticker} due to preprocessing failure.")
            continue

        # Create moving average spreads
        data = create_moving_average_spread(data, config['short_window'], config['long_window'])
        if data.empty:
            print(f"Skipping {ticker} due to spread creation failure.")
            continue

        # Feature engineering
        data = feature_engineering(data)
        if data.empty:
            print(f"Skipping {ticker} due to feature engineering failure.")
            continue

        # Autoencoder training
        data, scaler = train_autoencoder(data, autoencoder_features, config['encoding_dim'], config['autoencoder_epochs'], config['autoencoder_batch_size'])
        if scaler is None:
            print(f"Skipping {ticker} due to Autoencoder training failure.")
            continue

        # Prepare data for models
        X_train_seq, X_test_seq, X_train_flat, X_test_flat, y_train, y_test, feature_scaler, target_scaler = prepare_data_for_models(data, model_features, lookback=config['lookback'])
        if X_train_seq is None:
            print(f"Skipping {ticker} due to data preparation failure.")
            continue

        # Train models
        models = train_models(X_train_seq, X_train_flat, y_train, (X_train_seq.shape[1], X_train_seq.shape[2]), config['model_epochs'], config['model_batch_size'])
        if not models:
            print(f"Skipping {ticker} due to model training failure.")
            continue

        # Generate predictions and signals
        market_results = {}
        market_metrics = {}
        for model_name, model in models.items():
            try:
                if model_name in ["LSTM", "CNN-LSTM"]:
                    pred = model.predict(X_test_seq, verbose=0)
                else:
                    X_test = X_test_flat if model_name != "OLS" else sm.add_constant(X_test_flat)
                    pred = model.predict(X_test)

                signals = generate_signals(data, pred, target_scaler, entry_threshold=config['entry_threshold'], exit_threshold=config['exit_threshold'])
                if signals.empty:
                    print(f"Skipping {model_name} for {ticker} due to signal generation failure.")
                    continue

                # Backtest
                results = backtest_strategy(signals)
                if results.empty:
                    print(f"Skipping {model_name} for {ticker} due to backtesting failure.")
                    continue

                # Apply risk management
                results = apply_risk_management(results, config['max_position'], config['stop_loss'])
                if results.empty:
                    print(f"Skipping {model_name} for {ticker} due to risk management failure.")
                    continue

                # Evaluate performance
                metrics = evaluate_performance(results)
                market_results[model_name] = results
                market_metrics[model_name] = metrics

            except Exception as e:
                print(f"Error processing {model_name} for {ticker}: {e}")
                continue

        all_results[ticker] = market_results
        all_metrics[ticker] = market_metrics

    # Comparative Analysis Across Markets
    print("\n=== Comparative Analysis Across Markets ===")
    for ticker in all_metrics:
        print(f"\nMarket: {ticker}")
        if all_metrics[ticker]:
            comparison_df = pd.DataFrame(all_metrics[ticker]).T
            print(comparison_df)

            # Plot cumulative returns for this market
            plt.figure(figsize=(12, 6))
            for model_name, result in all_metrics[ticker].items():
                results_df = all_results[ticker].get(model_name)
                if results_df is not None and not results_df.empty:
                    plt.plot(results_df.index, results_df['Cumulative_Returns'], label=model_name)
            plt.title(f'Cumulative Returns Comparison - {ticker}')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No results available.")

# Run the pipeline
run_pipeline(CONFIG)