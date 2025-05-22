# WQU Capstone Project: Enhancing pair trading strategies for currency pairs using Deep Learning models: A focus on synthetic spread prediction

# Problem Statement
This capstone project investigates the application of deep learning techniques to enhance traditional pair trading strategies for the EUR/USD currency pair, with a specific focus on modeling and predicting a synthetic spread. Traditional pair trading methods, based on cointegration and mean reversion, often struggle with the nonlinear dynamics and structural shifts prevalent in forex markets. By developing a framework that utilizes advanced neural network models such as Long Short-Term Memory (LSTM), Convolutional Neural Network-Long Short-Term Memory (CNN-LSTM), and Autoencoders, this project aims to capture hidden temporal patterns and nonlinear dependencies in a synthetic EUR/USD spread. The research seeks to answer whether these advanced models can improve the profitability and resilience of long/short pair trading strategies. Through systematic data acquisition, feature engineering, model development, signal generation, and backtesting, this study demonstrates the potential of machine learning to address the limitations of conventional statistical methods in volatile financial markets. The findings could provide valuable insights into the practical application of AI in quantitative finance.

# Motivation for our project - Initial analysis
(The full code for initial analysis can be found on the `Motivation and initial analysis.ipynb` file)

Pair trading is an extremely popular market-neutral strategy that profits from short-term deviations in the prices of historically co-moving assets. The traditional statistical models used for the investigation of such opportunities in these holds are cointegration with mean-reversion. However, most of the approaches collapse under volatile and non-linear market conditions.

Understanding the underlying behavior of the selected market would help design a machine learning long/short pair trading strategy. The primary asset for our analysis in this project is the currency pair EUR/USD. Best evidence needs to be presented by which a long/short strategy is market-neutral.

So we did a thorough market analysis spanning a decade between 2014 and 2023, digging into the market structure to determine whether EUR/USD mainly consists of uptrends, downtrends, or horizontal trends.
```
Yearly Market Regime Summary (% Time in Each Regime):
Market_Type  Downtrend  Sideways  Uptrend
2014             23.75     76.25     0.00
2015              6.90     82.76    10.34
2016             15.33     84.67     0.00
2017              0.00     77.13    22.87
2018             23.75     76.25     0.00
2019             23.46     76.54     0.00
2020              0.00     75.95    24.05
2021             23.75     76.25     0.00
2022             21.92     78.08     0.00
2023             21.54     78.46     0.00
```

Preliminary market structure analysis shows that EUR/USD exhibits a predominantly range-bound behavior over the last 10 years.
* Sideways/Range-Bound Behavior: On average, 76% to 85% of the time, the EUR/USD market remained range-bound (non-directional) across all years.
* Limited Uptrend Phases: Notable uptrends were observed only in 2015, 2017, and 2020, and even then, they were relatively limited.
* Frequent Downtrends: Some years showed moderate periods of downtrends (e.g., 2014, 2018, 2019, 2022), but they did not dominate the market behavior.

The empirical evidence clearly indicates that the EUR/USD market predominantly operates in a sideways or non-directional regime over long periods. This market behavior makes trend-following strategies less effective and validates the choice of pursuing a market-neutral approach.

Given the range-bound nature of EUR/USD:
* Long/Short Pair Trading Strategies that exploit mean-reversion characteristics are highly appropriate.
* Machine Learning and Deep Learning models can be leveraged to better predict short-term divergences between asset pairs and generate profitable long/short trade signals.
* Focusing on building a data-driven, market-neutral
trading system based on these insights ensures alignment with the real-world behavior of the asset and increases the potential for success.

Thus, our project selection — applying machine/deep learning techniques to long/short pair trading strategies — is not only theoretically justified but also grounded in a careful empirical analysis of the target market.

# Workflow
This is the current workflow of the project. We implemented a demonstration of the full workflow in the Python file `MScFE_690_Team_9203_Robust.py`. The preliminary baseline results for Machine Learning models is in the IPython Notebook file.

![WhatsApp Image 2025-04-26 at 15 01 26_b617e48b](https://github.com/user-attachments/assets/9f9d6be7-7234-4d38-a747-3de9dcc8b75c)


# Code
The code for this project is organized into several sections, each corresponding to a key phase of the workflow. Below, we provide a detailed overview of the work done in each group of sections, along with result tables and visualizations where applicable. The full code for analysis and preliminary results is available in the committed IPython Notebook file, also accessible on Google Colab [here](https://colab.research.google.com/drive/1q8vL-H1VYMqL6DmmFG6h_N2LVgcWXgGj?usp=sharing).

## Sections 1-5: Data Acquisition and Preprocessing
In these sections, we collected and preprocessed historical EUR/USD price data spanning January 1, 2014, to January 1, 2024, using the `yfinance` library. The dataset comprises 2,381 trading days. We engineered features including moving averages (50-day and 200-day), technical indicators (RSI, MACD), volatility measures (ATR), and a synthetic spread defined as the difference between the closing price and its 200-day SMA. Stationarity of the synthetic spread was verified with the Augmented Dickey-Fuller (ADF) test (p-value < 0.05).

**Key Preprocessing Statistics**:

- **Dataset Size**: 2,381 observations
- **Feature Ranges**:
  - 50-day SMA: 0.9632 to 1.3721
  - 200-day SMA: 1.0023 to 1.3428
  - RSI (14-day): 21.34 to 78.92 (mean: 49.87, std: 14.23)
  - ATR (14-day): mean 0.0078, peak 0.0152
  - Synthetic Spread: mean -0.0002, std 0.0371, range -0.1423 to 0.1594

These steps ensured a clean, robust dataset with no significant missing values, critical for training deep learning models effectively.

## Sections 6-7: Model Development and Hyperparameter Tuning
We implemented and tuned several models to predict the synthetic spread:

- **LSTM-based Autoencoder**: Used for anomaly detection, optimized with 100 units, learning rate 0.001, and 50 epochs (validation MSE: 0.0032).
- **LSTM**: Two-layer architecture with 64 units each, trained for 50 epochs (test MSE: 0.41).
- **CNN-LSTM**: Integrated a 1D convolutional layer (32 filters) with an LSTM layer (64 units), yielding a test MSE of 0.43.
- **Random Forest**: Achieved MSE 0.3959, RMSE 0.6292, MAE 0.4698, $R^2$ 0.8018.
- **XGBoost**: Recorded MSE 0.4340, RMSE 0.6588, MAE 0.4873, $R^2$ 0.7827.
- **OLS**: Baseline model with MSE 0.4521, $R^2$ 0.7712.

**Model Performance Table**:

| Model         | MSE    | RMSE   | MAE    | $R^2$ |
|---------------|--------|--------|--------|-----------|
| LSTM          | 0.41   | 0.6403 | -      | -         |
| CNN-LSTM      | 0.43   | 0.6547 | -      | -         |
| Random Forest | 0.3959 | 0.6292 | 0.4698 | 0.8018    |
| XGBoost       | 0.4340 | 0.6588 | 0.4873 | 0.7827    |
| OLS           | 0.4521 | 0.6721 | -      | 0.7712    |

The Random Forest model showed superior predictive accuracy, while LSTM-based models excelled at capturing temporal dependencies, aligning with the project's focus on nonlinear dynamics.

## Sections 8-10: Signal Generation and Execution
Trading signals were derived from predicted Z-scores of the synthetic spread. The ensemble approach, averaging predictions from all models, generated 29 trades over the test period (position counts: {0: 2111, -1: 138, 1: 112}). The autoencoder's anomaly detection (95th percentile reconstruction error: 0.0045) filtered out two trades, reducing the total to 27.

**Signal Statistics**:

- **Ensemble Trade Frequency**: 29 trades (27 post-anomaly filtering)
- **Position Distribution**: 250 non-zero position days out of 2,381
- **Z-score Metrics**: Mean -0.0023, std 0.9871

This conservative trading approach reflects the ensemble's ability to balance signal quality and stability.

## Sections 11-13: Backtesting and Risk Management
Backtesting employed a walk-forward method with five folds, incorporating risk controls (max position: 1, 2% stop-loss per trade). The ensemble strategy executed 27 trades, yielding a cumulative return of 1.0111 (1.11% gain). The LSTM model outperformed with a return of 1.1037 (10.37%), while XGBoost lagged at 0.9944 (-0.56%).

**Backtesting Results**:

| Model    | Cumulative Return | Mean Daily Return | Std Dev  | Max Drawdown |
|----------|-------------------|-------------------|----------|--------------|
| Ensemble | 1.0111            | $5.12 \times 10^{-6}$ | 0.00087 | -0.0320      |
| LSTM     | 1.1037            | -                 | -        | -0.0744      |
| XGBoost  | 0.9944            | -                 | -        | -0.1381      |

The ensemble's low drawdown underscores its robustness, while the LSTM's higher returns highlight its potential in volatile markets.

## Section 14: Performance Evaluation and Visualization
The ensemble strategy was benchmarked against a buy-and-hold approach, which returned 0.8904 (-10.96%). The ensemble achieved a cumulative return of 0.0111, Sharpe ratio of 0.0865, and maximum drawdown of -0.0320, outperforming the benchmark in risk-adjusted terms.

**Performance Metrics Table**:

| **Model**       | **Cum. Return** | **Sharpe** | **Sortino** | **Calmar** | **Max Drawdown** |
|-----------------|-----------------|------------|-------------|------------|------------------|
| Buy-and-Hold    | -0.1096         | -0.1113    | -0.1112     | 0.0519     | -0.2351          |
| LSTM            | 0.1037          | 0.3366     | 0.1683      | 0.1424     | -0.0744          |
| CNN-LSTM        | -0.0307         | -0.0943    | -0.0428     | 0.0470     | -0.0705          |
| RandomForest    | 0.0539          | 0.1475     | 0.1388      | 0.0276     | -0.2038          |
| XGBoost         | -0.0056         | -0.0027    | -0.0016     | 0.0043     | -0.1381          |
| OLS             | 0.0551          | 0.1508     | 0.1337      | 0.0273     | -0.2104          |
| Ensemble        | 0.0111          | 0.0865     | 0.0203      | 0.0366     | -0.0320          |

**Visualization**:
The image below compares the cumulative returns of the ensemble strategy and buy-and-hold strategy from 2015 to 2024. The ensemble strategy (blue line) remains stable near 1.00, while the buy-and-hold strategy (orange line) fluctuates significantly between 0.85 and 0.99.

![Ensemble Performance vs Buy-and-Hold](https://github.com/user-attachments/assets/b26bf2a2-fc44-4bb6-a1af-7b58257c4214)
*Figure: Comparison of cumulative returns for the ensemble strategy and buy-and-hold strategy over 2015-2024.*

This graph illustrates the ensemble strategy's ability to maintain consistent performance, reinforcing the value of deep learning in enhancing pair trading strategies.
