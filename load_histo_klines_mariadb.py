import json
import pandas as pd
from sqlalchemy import create_engine, text, Column, MetaData, String, Float, DateTime, Table
import os

# Define the connection URL
DB_USERNAME = os.environ.get('USER')
DB_PASSWORD = os.environ.get('MARIADB_PW')
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'ProjetOPA'

# Create the connection URL
connection_url = f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Create the database engine
engine = create_engine(connection_url)

# Connect to the database
connection = engine.connect()

# Convert json file to DataFrame
df = pd.read_json('crypto_histo.json', orient = 'split', compression = 'infer')

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
metadata.create_all(engine)

# Write the DataFrame to the database
df.to_sql('data_market', con=engine, if_exists='append', index=False)

#Vérification
print('Data loaded in data_market table.')
# Close the connection
connection.close()