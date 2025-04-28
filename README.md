# WQU Capstone Project: Enhancing Pair Trading Strategies for Currency Pairs Using Deep Learning Models

# Problem Statement
To identify opportunities, traditional pair trading strategies often rely on statistical tools like cointegration and mean reversion. These methods assume linear relationships and stable dynamics, but markets—especially the fast-moving forex market—frequently exhibit nonlinear behaviors and structural shifts that challenge those assumptions. Our project aims to develop a deep learning-driven framework to detect and exploit short-term deviations in a synthetic EUR/USD and other currency pairs’ spread, utilizing neural networks to uncover hidden temporal patterns and nonlinear dynamics, thereby enhancing the profitability and resilience of long/short pair trading strategies.

# Workflow
![WhatsApp Image 2025-04-26 at 15 01 26_b617e48b](https://github.com/user-attachments/assets/9f9d6be7-7234-4d38-a747-3de9dcc8b75c)


# Code
The code is currently in the committed IPython Notebook file. The file on Google Colab is also accessible [here](https://colab.research.google.com/drive/1TctfiBYEhWehlVWe3qZgFIM5_EYYM4CS?usp=sharing)

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
