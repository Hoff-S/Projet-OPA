from fastapi import FastAPI
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine, Column, MetaData, String, DateTime, Table, insert, select, func
import os
from pydantic import BaseModel
from datetime import datetime
import requests
import numpy as np
import mlflow
from datetime import timedelta

# MLFLOW TRACKING URI
if not os.environ.get('MLFLOW_TRACKING_URI'):
    ip = requests.get('https://api.ipify.org').content.decode('utf8')
    uri= 'http://{}'.format(ip) + ':8085'
    mlflow_tracking_uri = uri
else:
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')

mlflow.set_tracking_uri(mlflow_tracking_uri)

# API
api = FastAPI(
    title='Cryptobot'
)

# Access to the mongodb
username = "admin"
password = os.environ.get('MONGODB_PW')
host = os.environ.get('MONGODB_HOST')

client = MongoClient('mongodb://%s:%s@%s:27017' % (username, password, host))

db = client['cryptobot']

col_name = "klines"
collection = db[col_name]

# creating a Kline class
class Kline(BaseModel):
    open_time: datetime
    symbol : str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float

# Access to mariadb
DB_USERNAME = "root"
DB_PASSWORD = os.environ.get('MARIADB_PW')
DB_HOST = os.environ.get('MARIADB_HOST')
DB_PORT = '3306'
DB_NAME = 'ProjetOPA'

# Create the connection URL
connection_url = f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Create the database engine
engine = create_engine(connection_url)

# Connect to the database
connection = engine.connect()

metadata = MetaData()
fastapi_logs = Table(
        'fastapi_logs',
        metadata,
        Column('log_time', DateTime),
        Column('log_level', String(10)),
        Column('log_message', String(50))
        )

if not engine.dialect.has_table(connection, table_name='fastapi_logs'):
    fastapi_logs.create(engine)

def log_message(log_time, log_level, log_message):
    stmt = insert(fastapi_logs).values(log_time=log_time, log_level=log_level, log_message=log_message)
            
    with engine.connect() as connection:
        result = connection.execute(stmt)
        connection.commit()

def fastapi_statistics():
    stmt = select(func.date(fastapi_logs.c.log_time), func.substring_index(func.replace(fastapi_logs.c.log_message," ","/"),"/",2), func.count(1).label("count"))\
            .where(fastapi_logs.c.log_message.like('/%') )\
            .where(fastapi_logs.c.log_level == 'DEBUG')\
            .group_by(func.date(fastapi_logs.c.log_time), func.substring_index(func.replace(fastapi_logs.c.log_message," ","/"),"/",2))\
            .order_by(func.date(fastapi_logs.c.log_time).asc(), func.substring_index(func.replace(fastapi_logs.c.log_message," ","/"),"/",2).asc())
            
    with engine.connect() as connection:
        results = connection.execute(stmt)
    
    rows = results.fetchall()

    dict_rows = [{'date' : row[0], 'route' : row[1], 'count' : row[2]} for row in rows]

    return dict_rows

def invboxcox(y, lmbda):
    """Fonction inverse de la transformée de Box-Cox"""
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lmbda * y + 1) / lmbda)
    
def get_experiment_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
        print(f"Experiment ID: {experiment_id}")
        return experiment_id
    else:
        print(f"L'expérience '{experiment_name}' n'existe pas.")
        return None

def get_active_run_id(experiment_id):

    """Return the active run id from MLFLOW server, for a given experiment ID"""

    # Search active run
    r = requests.post(f'{mlflow_tracking_uri}/api/2.0/mlflow/runs/search',
                  json={'experiment_ids': [experiment_id], 'max_results':1}
                  )

    # Active run info from the MLFLOW Server
    data = r.json()

    # Get active run_id
    run_id = data['runs'][0]['info']['run_id']
    print(f"Run ID: {run_id}")

    return run_id

def fetch_prediction():

    # Experiment ID
    experiment_id = get_experiment_id("Crypto_Models")
    if experiment_id is None:
        return None

    # Run ID
    run_id = get_active_run_id(experiment_id=experiment_id)

    # Charger les paramètres du modèle
    run = mlflow.get_run(run_id)
    lmbda = float(run.data.params['lambda'])
    
    model_uri = f'runs:/{run_id}/model'
    loaded_model = mlflow.statsmodels.load_model(model_uri)

    # Obtenir la date actuelle
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Faire des prédictions pour demain avec le modèle chargé
    steps = 1
    forecast = loaded_model.get_forecast(steps=steps)
    forecast_dates = [current_date + timedelta(days=i) for i in range(steps)]
    predictions = forecast.summary_frame()
    predictions.index = forecast_dates

    # Appliquer l'inverse de la transformation Box-Cox
    predictions['mean'] = invboxcox(predictions['mean'], lmbda)
    predictions['mean_ci_lower'] = invboxcox(predictions['mean_ci_lower'], lmbda)
    predictions['mean_ci_upper'] = invboxcox(predictions['mean_ci_upper'], lmbda)

    result = {
        "Prediction": predictions['mean'].iloc[0],
        "Borne inférieure de l'intervalle de confiance": predictions['mean_ci_lower'].iloc[0],
        "Borne supérieure de l'intervalle de confiance": predictions['mean_ci_upper'].iloc[0]
    }

    return result

# API status
@api.get('/status')
async def get_status():
    '''
    Returns Ready
    '''
    log_message(datetime.now(),"DEBUG","/status")

    return {'message':'Ready'}

# API last
@api.get('/last/{symbol:str}')
async def get_last(symbol):
    '''
    Returns last kline for a symbol
    '''
    log_message(datetime.now(),"DEBUG",f"/last/{symbol}")

    results = collection.find({'symbol':symbol},{'open_time':1, 'symbol':1, 
                                                  'open_price':1, 'high_price':1,
                                                  'low_price':1, 'close_price':1,
                                                   'volume':1, '_id':0}).sort('_id', -1)[0]

    return [results]

# API all
@api.get('/all/{symbol:str}')
async def get_all(symbol):
    '''
    Returns all klines for a symbol
    '''
    log_message(datetime.now(),"DEBUG",f"/all/{symbol}")

    results = collection.find({'symbol':symbol},{'open_time':1, 'symbol':1, 
                                                  'open_price':1, 'high_price':1,
                                                  'low_price':1, 'close_price':1,
                                                   'volume':1, '_id':0})

    res = [row for row in results] 
    return res

# API insert
@api.post('/insert')
def post_insert(kline : Kline):
    '''
    Returns Added
    '''
    log_message(datetime.now(),"DEBUG","/insert")

    collection.insert_one(dict(kline))

    return {'message':'Added'}

@api.get('/statistics')
async def get_statistics():
    '''
    Returns statistics per day and per route
    '''
    log_message(datetime.now(),"DEBUG","/statistics")
    results = fastapi_statistics()

    return results

@api.get('/prediction')
async def get_prediction():
    '''
    Returns tomorrow close_price prediction
    '''
    log_message(datetime.now(),"DEBUG","/prediction")
    results = fetch_prediction()
    
    return results
