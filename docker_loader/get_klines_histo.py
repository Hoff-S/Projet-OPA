import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import sys
from pymongo import MongoClient
import os
from sqlalchemy import create_engine, text, Column, MetaData, String, Float, DateTime, Table, func, select
    

def prepare_data(date_toget, symbol_toget):
    
    # Use the binance API
    lst = []
    for symbol in symbol_toget:

        klinesT = Client().get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, date_toget)
        df = pd.DataFrame(klinesT, columns = ['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        
        # Clean dataset
        df = df.drop(['ignore', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av'], axis =1)

        df.insert(1,'symbol',symbol) 

        df['close_price'] = pd.to_numeric(df['close_price'])
        df['high_price'] = pd.to_numeric(df['high_price'])
        df['low_price'] = pd.to_numeric(df['low_price'])
        df['open_price'] = pd.to_numeric(df['open_price'])
        df['volume'] = pd.to_numeric(df['volume'])

        # Convert time (ms) to datetime

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        lst.append(df)
    
    df = pd.concat(lst, ignore_index=True)
    
    return df


def load_mongo(df):
    username = "admin"
    password = os.environ.get('MONGODB_PW')
    host = os.environ.get('MONGODB_HOST')

    client = MongoClient('mongodb://%s:%s@%s:27017' % (username, password, host))
    db = client['cryptobot']

    col_name = "klines"
    
    # delete the collection if exists

    collist = db.list_collection_names()
    if col_name in collist:
        collection = db[col_name]
        collection.drop()
        print("The {} collection exists. It has been deleted and created.".format(col_name))
    else:
        print("The {} Collection is created.".format(col_name))
        collection = db[col_name]

    data = df.to_dict('records')

    # load the data into the mongodb collection
    
    collection.insert_many(data)

    data_size = collection.count_documents({})

    print("Data loaded ({}) in {} collection.".format(data_size, col_name))
    
    # close the connection
    client.close()

def load_mariadb(df):
    

    # Define the connection URL
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

    # Use the DataFrame
    data = df
    
    # Define metadata of tables
    # data_market table
    metadata = MetaData()

    data_market = Table(
        'data_market',
        metadata,
        Column('open_time', DateTime, primary_key=True),
        Column('symbol', String(20), primary_key=True),
        Column('open_price', Float),
        Column('high_price', Float),
        Column('low_price', Float),
        Column('close_price', Float),
        Column('volume', Float)
    )

    # Create the table in the database
    if engine.dialect.has_table(connection, table_name='data_market'):
        data_market.drop(engine)
        data_market.create(engine)
        print("The data_market table exists. It has been deleted and created.")
    else:
        data_market.create(engine)
        print("The data_market table is created.")

    # Write the DataFrame to the database
    data.to_sql('data_market', con=engine, if_exists='append', index=False)

    stmt=select(func.count(1)).select_from(text('data_market'))
    res = connection.execute(stmt)
    for row in res:
        data_size= row[0]

    #Vérification
    print('Data loaded ({}) in data_market table.'.format(data_size))
    # Close the connection
    connection.close()


# Run the functions
df_klines = prepare_data("01 January 2022", ["BTCEUR","ETHEUR","BCHEUR","SOLEUR", "ADAEUR"])
load_mongo(df_klines)
load_mariadb(df_klines)
