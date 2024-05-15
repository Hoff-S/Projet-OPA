import pandas as pd
from binance.client import Client
from sqlalchemy import create_engine, text

# Check if data already exists in database before loading it

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

# Create connection
connection = engine.connect()

def check_existing_data(symbol, date):
    """Check if data already exists in the database."""
    query = f"SELECT COUNT(*) FROM data_market WHERE (symbol = '{symbol}') AND (open_time = '{date}')"
    result = connection.execute(text(query))
    count = result.fetchone()[0]
    return count > 0

# Convert the crypto_histo.json file to a DataFrame
df = pd.read_json('crypto_histo.json', orient = 'split', compression = 'infer')

def load_data(df, db_url, table):
    """Load the DataFrame DF in the table, if each row of the DF doesn't already exists in the table

    Args:
        df (DataFrame): DataFrame to load to the table
        db_url (str): url of the database
        table (str): table name
    """
    for index in range(len(df)):
        row = df.iloc[index]
        if check_existing_data(row['symbol'], row['open_time']): 
            print(f"Data already exists in {table} for symbol {row['symbol']} and open_time {row['open_time']}.")
        else:     
            # Extract the row as a DF
            row_df = pd.DataFrame([row])
            # Load the row in the table
            row_df.to_sql(name=table, con=engine, if_exists='append', index=False)
            print(f"Data for symbol {row['symbol']} and open_time {row['open_time']} successfully loaded into {table}.")

load_data(df, connection_url, 'data_market')

connection.close()

