import json
import pandas as pd
from sqlalchemy import create_engine, text

# Define the connection URL
DB_USERNAME = 'user'
DB_PASSWORD = 'secretpw'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'ProjetOPA'

# Create the connection URL
connection_url = f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Create the database engine
engine = create_engine(connection_url)

# Connect to the database
connection = engine.connect()

#Convert json file to DataFrame
df = pd.read_json('crypto_histo.json', orient = 'split', compression = 'infer')

# Write the DataFrame to the database
df.to_sql('data_market', con=engine, if_exists='replace', index=False)

# Close the connection
connection.close()