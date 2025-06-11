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
The code for this project is organized into several sections, each corresponding to a key phase of the workflow. Below, we provide a detailed overview of the work done in each group of sections, along with result tables and visualizations where applicable. The full code for analysis and results is available in the committed IPython Notebook file, also accessible on Google Colab [here](https://colab.research.google.com/drive/1q8vL-H1VYMqL6DmmFG6h_N2LVgcWXgGj?usp=sharing).

### Data Acquisition and Preparation
We began by setting up the computational environment with essential Python libraries, including `pandas`, `numpy`, `yfinance`, `statsmodels`, `sklearn`, `xgboost`, `tensorflow`, `matplotlib`, `seaborn`, and `optuna`. A centralized configuration dictionary, `CONFIG`, was defined to manage parameters such as the market ticker (`EURUSD=X`), data range (2014-01-01 to 2024-12-31), and feature engineering windows.

Daily EUR/USD price data was fetched using `yfinance`, resulting in a dataset with columns for date, open, high, low, and close prices. Custom utility functions ensured data robustness: `get_market_data` cleaned the data by handling missing values via forward-fill imputation, while functions like `frac_diff` computed fractionally differentiated series, and others prepared technical indicators.

### Feature Engineering and Analysis
A synthetic spread was created by subtracting a 50-day moving average from the closing price. The dataset was enriched with features including log returns, lagged returns (lags 1–5), RSI (14-day), MACD (12- and 26-day), Bollinger Bands (20-day, 2 std), ATR (14-day), Garman-Klass volatility (20-day), and a fractionally differentiated spread (\(d=0.5\), 252-day window).

Correlation analysis removed features with correlations above 0.85, preserving critical variables (see Figure 1). Stationarity of the synthetic spread was validated using fractional differentiation, confirmed by ADF tests (see Figure 2).

**Figure 1**: Feature Correlation Heatmap Before and After Pruning  
![image](https://github.com/user-attachments/assets/657dacfd-7aee-4b7d-a24d-2749a7d28320)
![image](https://github.com/user-attachments/assets/c4fd132f-48cc-4e31-b104-8ca052b41bed)


**Figure 2**: ADF Test and Distribution Comparison for Original and Fractionally Differentiated Spread  
![image](https://github.com/user-attachments/assets/41029d69-6495-4f8a-9315-35abf92a7ca8)
![image](https://github.com/user-attachments/assets/37293ab0-4917-46e3-96da-ef618753afd8)


### Model Development and Tuning
We developed an ensemble of models to predict the fractionally differentiated spread:
- **Deep Learning Models**: LSTM, LSTM with attention, CNN-LSTM with attention, and an LSTM-based autoencoder.
- **Traditional Models**: Random Forest, XGBoost, and OLS regression.

Hyperparameter tuning was performed using `Optuna` on an initial training set (50% of data). Deep learning models optimized parameters like LSTM units (32–128), learning rates (10⁻⁴–10⁻²), filters (16–64), and kernel sizes (2–5). Traditional models tuned estimators (50–300), depths (5–20 for RF, 3–10 for XGB), and learning rates (10⁻³–0.3 for XGB).

### Strategy Execution and Backtesting
A walk-forward validation approach with 10 folds (100-day test windows) simulated real-time trading. Predictions from all models were averaged into an ensemble, weighted dynamically based on inverse MAE after three folds. Trading signals were derived using Z-score thresholds (entry: 1.5, exit: 0.5) with RSI filters. Risk management included stop-losses (5%), take-profits (10%), and Kelly criterion position sizing (fraction 0.2), accounting for transaction costs (0.02%).

### Final Performance Analysis, Comparison, & Visualization
In this final step, we evaluated the trading strategy using key metrics: cumulative returns, Sharpe ratio, Sortino ratio, Calmar ratio, and maximum drawdown. The ensemble strategy was compared against individual models and a buy-and-hold benchmark.

**Predictive Power Analysis**:

The table below summarizes the predictive accuracy of each model, measured by average MAE, RMSE, R-squared, and the standard deviation of R-squared across folds.

| Model              |   Avg MAE |   Avg RMSE |   Avg R-Squared |   Std Dev (R-Squared) |
|:-------------------|----------:|-----------:|----------------:|----------------------:|
| OLS                |    0.0015 |     0.0018 |          0.896  |                0.0825 |
| XGB                |    0.0016 |     0.0019 |          0.8919 |                0.0622 |
| RF                 |    0.0017 |     0.002  |          0.8816 |                0.0589 |
| LSTM               |    0.0041 |     0.0053 |          0.028  |                0.1623 |
| ENSEMBLE           |    0.0065 |     0.0091 |        -11.5474 |               32.1661 |
| CNN_LSTM_ATTENTION |    0.0427 |     0.0524 |       -148.137  |              209.094  |
| LSTM_ATTENTION     |    0.1741 |     0.1978 |      -1750.05   |             2023.36   |

**Financial Performance Analysis**:

The following table presents the financial performance of each model's trading strategy, including cumulative return, risk-adjusted ratios, and maximum drawdown.

| Model              |   Cumulative Return (%) |   Sharpe Ratio |   Sortino Ratio |   Calmar Ratio |   Max Drawdown (%) |
|:-------------------|------------------------:|---------------:|----------------:|---------------:|-------------------:|
| CNN_LSTM_ATTENTION |                0        |        0       |        0        |       0        |           0        |
| LSTM               |                0        |        0       |        0        |       0        |           0        |
| LSTM_ATTENTION     |                0        |        0       |        0        |       0        |           0        |
| ENSEMBLE           |               -0.786978 |       -3.38388 |       -0.770235 |      -0.252745 |          -0.786978 |
| OLS                |               -0.898032 |       -3.68757 |       -0.917066 |      -0.252851 |          -0.898032 |
| RF                 |               -1.50367  |       -3.89715 |       -1.09785  |      -0.25343  |          -1.50367  |
| XGB                |               -1.50367  |       -3.89715 |       -1.09785  |      -0.25343  |          -1.50367  |

**Ensemble Comparison**:

We also compared the all-model ensemble with a top 3 ensemble (RF, XGB, OLS):

**Predictive Power Comparison**:

| Metric        |   All-Model Ensemble |   Top 3 Ensemble |
|:--------------|---------------------:|-----------------:|
| Avg MAE       |               0.0065 |           0.0015 |
| Avg RMSE      |               0.0091 |           0.0017 |
| Avg R-Squared |             -11.5474 |           0.9067 |

**Financial Performance Comparison**:

| Metric                |   All-Model Ensemble |   Top 3 Ensemble |
|:----------------------|---------------------:|-----------------:|
| Cumulative Return (%) |              -0.787  |          -1.0667 |
| Sharpe Ratio          |              -3.3839 |          -3.9182 |
| Sortino Ratio         |              -0.7702 |          -1.022  |
| Calmar Ratio          |              -0.2527 |          -0.253  |
| Max Drawdown (%)      |              -0.787  |          -1.0667 |

**Key Insights**:
- **Predictive Accuracy**: Traditional models (OLS, XGB, RF) significantly outperformed deep learning models, achieving low errors and high R-squared values (e.g., OLS: MAE 0.0015, R² 0.896).
- **Financial Performance**: All strategies incurred losses, with the all-model ensemble performing best among them (cumulative return -0.787%, Sharpe -3.3839). The buy-and-hold benchmark yielded approximately -10%.
- **Ensemble Comparison**: The top 3 ensemble showed superior predictive power (R² 0.9067 vs. -11.5474) but worse financial performance (-1.0667% vs. -0.787%), highlighting the challenge of translating predictive accuracy into profitability with daily data.

**Figure 3**: Model Comparison by Average Mean Absolute Error (MAE)  
![image](https://github.com/user-attachments/assets/9852afa6-5c11-473a-bcb8-633ef3e2d278)


This analysis underscores the limitations of daily data in capturing short-lived mean-reverting opportunities, suggesting that higher-frequency data could better leverage the models' predictive capabilities.
