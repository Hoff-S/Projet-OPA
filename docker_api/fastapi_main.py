from fastapi import FastAPI
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine, Column, MetaData, String, DateTime, Table, insert, select, func
import os
from pydantic import BaseModel
from datetime import datetime

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