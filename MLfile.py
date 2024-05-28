import numpy as np
import pandas as pd
import os
from sqlalchemy import create_engine, text
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from itertools import product
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

# -----------------------------------------------------------------------------------------------
### GET DATA FROM DATABASE

# Define the connection URL
DB_USERNAME = os.environ.get('USER')
DB_PASSWORD = os.environ.get('MARIADB_PW')
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'ProjetOPA'

# Create the connection URL
connection_url = f'mysql+pymysql://user:secretpw@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Create the database engine
engine = create_engine(connection_url)

# Create connection
connection = engine.connect()

# Get data to DF
query = 'SELECT * FROM data_market'
result = connection.execute(text(query))
df = pd.DataFrame(result)

# Close connection
connection.close()

# -----------------------------------------------------------------------------------------------
### DATA PREPROCESSING

# Set DataFrame index
df = df.set_index('open_time')

# Convert to Series
close_price_series = pd.Series(df['close_price'])

# Resampling to daily frequency
df_day = df.drop('symbol', axis=1).resample('D').mean()

# Resampling to monthly frequency
df_month = df.drop('symbol', axis=1).resample('ME').mean()

# Resampling to annual frequency
df_year = df.drop('symbol', axis=1).resample('YE-DEC').mean()

# Resampling to quarterly frequency
df_Q = df.drop('symbol', axis=1).resample('QE-DEC').mean()

# -----------------------------------------------------------------------------------------------
### DATA VIZ

"""
fig = plt.figure(figsize=[15, 7])
plt.suptitle('Bitcoin exchanges, mean EUR', fontsize=22)

plt.subplot(221)
plt.plot(close_price_series, '-', label='By Days')
plt.legend()

plt.subplot(222)
plt.plot(close_price_series_month, '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(close_price_series_Q, '-', label='By Quarters')
plt.legend()

plt.subplot(224)
plt.plot(close_price_series_year, '-', label='By Years')
plt.legend()

# plt.tight_layout()
plt.show()
"""

# -----------------------------------------------------------------------------------------------
### SEASONAL DECOMPOSE

"""
sm.tsa.seasonal_decompose(close_price_series_day, period=7, model='multiplicative').plot()
plt.show()
"""

# -----------------------------------------------------------------------------------------------
### SERIE DIFFERENCIATION - note : peut être amélioré avec la transformation de box-cox

# Log Transform (not used here)
"""
# Rename working series
CPSD = close_price_series_day

# Log Transformation
CPSD_log = np.log(CPSD)

# Simple Differenciation
CPSD_log_1 = CPSD_log.diff().dropna()

# 7-Order Period Differenciation
CPSD_log_2 = CPSD_log_1.diff(periods=7).dropna()

# We determine if the 2-times differenciate serie is stationary with p_value
_, p_value, _, _, _, _ = sm.tsa.stattools.adfuller(CPSD_log_2)

if p_value < 0.05:
    print('We consider the 2-times differenciate series is stationary')
else:
    print('We consider the 2-times differenciate series is not stationary')
"""

# BoxCox Transformation
df_day['Weighted_Price_box'], lmbda = stats.boxcox(df_day['close_price'])
print("BoxCox Transform -> Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_day[['Weighted_Price_box']])[1])

# Seasonal Diffirenciation
df_day['prices_box_diff'] = df_day[['Weighted_Price_box']] - df_day[['Weighted_Price_box']].shift(7)
print("Seasonal Differenciation -> Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_day[['prices_box_diff']][7:])[1])

# Regular Differenciation
df_day['prices_box_diff2'] = df_day[['prices_box_diff']] - df_day[['prices_box_diff']].shift(1)
print("Regular Differenciation -> Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_day[['prices_box_diff2']][8:])[1])

# Seasonal decomposition
"""
sm.tsa.seasonal_decompose(df_day[['prices_box_diff2']][8:]).plot()   
plt.show()
"""

# -----------------------------------------------------------------------------------------------
### MODEL SELECTION

'''
# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(CPSD_log_2, lags=50, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(CPSD_log_2, lags=50, ax=ax)
plt.tight_layout()
plt.show()

print('We estimate firstly with ARMA(1,1)')
'''

# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(df_day[['Weighted_Price_box']], order=(param[0], D, param[1]), 
                                        seasonal_order=(param[2], d, param[3], 7)).fit(disp=-1)
        print(f'{param} -> Done')
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())

# Residues Analysis
'''
plt.figure(figsize=(15,7))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=50, ax=ax)

print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title('Residuals')
plt.show()
'''

# Coeff Stability
'''
plt.figure(figsize=(12, 6))
plt.plot(best_model.params, marker='o')
plt.title('Stabilité des coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Valeur')
plt.grid(True)
plt.show()
'''
# -----------------------------------------------------------------------------------------------
### PREDICTION

# Inverse Box-Cox Transformation Function
def invboxcox(y, lmbda):
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lmbda * y + 1) / lmbda)

df_day2 = df_day[['close_price']]
prediction = best_model.get_forecast(steps=20).summary_frame()
fig, ax = plt.subplots(figsize = (15,5))

plt.plot(df_day2)
prediction = invboxcox(prediction, lmbda)

prediction['mean'].plot(ax = ax, style = 'k--') #Visualisation de la moyenne

ax.fill_between(prediction.index, prediction['mean_ci_lower'], prediction['mean_ci_upper'], color='k', alpha=0.1)
plt.show()

# -----------------------------------------------------------------------------------------------
### CROSS TEST

# Split Data
train_size = int(len(df_day) * 0.80)
train_data, test_data = df_day[:train_size], df_day[train_size:]

# Model fitting on train set
best_model_train = sm.tsa.statespace.SARIMAX(train_data['Weighted_Price_box'], 
                                             order=(best_param[0], D, best_param[1]), 
                                             seasonal_order=(best_param[2], d, best_param[3], 7)).fit(disp=-1)

# Predictions on test set
predictions = best_model_train.get_forecast(steps=len(test_data))
predicted_values = predictions.predicted_mean
predicted_values = invboxcox(predicted_values, lmbda)

# MSE calculation
test_data['predicted_close_price'] = predicted_values
rmse = sqrt(mean_squared_error(test_data['close_price'], test_data['predicted_close_price']))
print(f"Root Mean Squared Error: {rmse}")

# Result viz
plt.figure(figsize=(15,7))
plt.plot(train_data.index, train_data['close_price'], label='Train')
plt.plot(test_data.index, test_data['close_price'], label='Test')
plt.plot(test_data.index, test_data['predicted_close_price'], label='Predicted')
plt.legend()
plt.show()
