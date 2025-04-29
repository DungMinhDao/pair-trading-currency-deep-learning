# WQU Capstone Project: Enhancing Pair Trading Strategies for Currency Pairs Using Deep Learning Models

# Problem Statement
This capstone project investigates the application of deep learning techniques to enhance traditional pair trading strategies for the EUR/USD currency pair and other currency pairs. Traditional pair trading methods, based on cointegration and mean reversion, often struggle with the nonlinear dynamics and structural shifts prevalent in forex markets. By developing a framework that utilizes advanced neural network models such as Long Short-Term Memory (LSTM), Convolutional Neural Network-Long Short-Term Memory (CNN-LSTM), and Autoencoders, this project aims to capture hidden temporal patterns and nonlinear dependencies in a synthetic EUR/USD spread. The research seeks to answer whether these advanced models can improve the profitability and resilience of long/short pair trading strategies. Through systematic data acquisition, feature engineering, model development, signal generation, and backtesting, this study demonstrates the potential of machine learning to address the limitations of conventional statistical methods in volatile financial markets. The findings could provide valuable insights into the practical application of AI in quantitative finance.

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
The code for analysis and preliminary results is currently in the committed IPython Notebook file. The file on Google Colab is also accessible [here](https://colab.research.google.com/drive/1TctfiBYEhWehlVWe3qZgFIM5_EYYM4CS?usp=sharing)

## Code structure explanation (notebook)

### Step 1: Data Acquisition
Data for currency pairs EUR/USD, GBP/USD, USD/JPY, USD/CHF, and EUR/GBP from 2015-01-01 to 2024-01-01 is acquired.

### Step 2: Exploratory Data Analysis
In this section, we visualize the Normalized Price for Currency Pairs (2015-2024) and the Correlation Matrix of Price for Currency Pairs.

![Normalized Price for Currency Pairs (2015-2024)](https://github.com/user-attachments/assets/dafb0b88-4b8e-4502-a8b2-d88f7a2a2386)
![Correlation Matrix of Price for Currency Pairs](https://github.com/user-attachments/assets/495c1168-4faf-497e-967c-d23266e4141f)

### Step 3: Feature Engineering
List of engineered features: `['RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'BB_High', 'BB_Low', 'BB_Width']` and `['EMA13', 'EMA34', 'EMA55', 'EMA13_above_34', 'EMA13_above_55', 'EMA34_above_55']`.

Targets: `Target_Z_Score`, `Target_Direction`

![EUR/USD Z-Score (Synthetic Spread)](https://github.com/user-attachments/assets/0c13d63f-dc07-43be-ab49-0727f5e06591)

### Step 4: Feature Selection and Data Preparation
We focus on EUR/USD for modeling. First, we drop columns we don't want to use as features: `drop_cols = ['Target_Z_Score', 'Target_Direction', 'Price', 'Volume']`

Training set size: (1827, 19), Test set size: (457, 19)

### Step 5: Model Training and Evaluation
#### 5.1: Random Forest Regression
Random Forest Performance Metrics:
* MSE: 0.395868
* RMSE: 0.629181
* MAE: 0.469814
* R²: 0.801765
#### 5.2: XGBoost Regression
XGBoost Performance Metrics:
* MSE: 0.433958
* RMSE: 0.658755
* MAE: 0.487309
* R²: 0.782692
#### 5.3: Basic Trading Strategy Simulation
![Trading Strategy Performance Comparison](https://github.com/user-attachments/assets/56363ea1-6c6b-44fd-b129-ff12fecb2a98)

### Step 6: Cross-validation for Time Series (To be updated)
