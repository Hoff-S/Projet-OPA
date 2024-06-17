import pandas as pd
import os
from sqlalchemy import create_engine, text
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
import mlflow
from mlflow.tracking import MlflowClient
import requests

# -----------------------------------------------------------------------------------------------
#Input

API_URL = os.environ.get('API_URL')
mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
symbol = 'BTCEUR'

# -----------------------------------------------------------------------------------------------
### GET DATA FROM API

url = f'http://{API_URL}:8000/all/{symbol}'
r = requests.get(url)
data = r.json()
df_day = pd.DataFrame(data)

# -----------------------------------------------------------------------------------------------
### DATA PREPROCESSING

# Set DataFrame index
df_day = df_day.set_index('open_time')

# -----------------------------------------------------------------------------------------------

# BoxCox Transformation
df_day['Weighted_Price_box'], lmbda = stats.boxcox(df_day['close_price'])
#print("BoxCox Transform -> Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_day[['Weighted_Price_box']])[1])

# Seasonal Diffirenciation
df_day['prices_box_diff'] = df_day[['Weighted_Price_box']] - df_day[['Weighted_Price_box']].shift(7)
#print("Seasonal Differenciation -> Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_day[['prices_box_diff']][7:])[1])

# Regular Differenciation
df_day['prices_box_diff2'] = df_day[['prices_box_diff']] - df_day[['prices_box_diff']].shift(1)
#print("Regular Differenciation -> Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_day[['prices_box_diff2']][8:])[1])

# -----------------------------------------------------------------------------------------------
### MODEL SELECTION

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

# -----------------------------------------------------------------------------------------------
### EXPORT TO MLFLOW

print(best_model)

client = MlflowClient(tracking_uri=mlflow_tracking_uri)

crypto_experiment = mlflow.set_experiment("Crypto_Models")
run_name = "first_run"
artifact_path = "sarima_crypto"

with mlflow.start_run(run_name=run_name) as run:

    # Trained model
    mlflow.sklearn.log_model(sk_model=best_model, artifact_path=artifact_path)

    # Save datas
    mlflow.log_table(data=df_day, artifact_file="data.json")

    # Save params
    mlflow.log_params({"lambda": lmbda, "order_p": best_param[0], "order_d": d, "order_q": best_param[1], "seasonal_order_P": best_param[2], "seasonal_order_D": D, "seasonal_order_Q": best_param[3]})

    # Save metrics
    mlflow.log_metrics({"yesterday_close_price_value": df_day['close_price'].iloc[-1]})