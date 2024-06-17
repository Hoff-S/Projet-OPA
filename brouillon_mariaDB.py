import pandas as pd
from binance.client import Client
from sqlalchemy import create_engine
import pymysql

#------------------------------------------------------------------------------------------------
# CREATE & CLEAN DF

# Define which symbol data market add to the database and since when
symbol = 'BTCEUR'
start_date = "01 January 2022"

def get_data(symbol, start_date):
    """Get data from Binance.

    Args:
        symbol (str): Symbol wanted from Binance (ex: BTCEUR, ETHUSD ...)
        start_date (str): Date the data has been collected since (ex: '01 January 2022')

    Returns:
        DataFrame: DF containing market data of the symbol since the input start_date
    """
    klinesT = Client().get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, start_date)
    df = pd.DataFrame(klinesT, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    Client().close_connection
    return df

#get data from Binance
df = get_data(symbol, start_date)


def clean_data(df):
    """Clean a dataframe from get_historical_klines or get_klines method, only keeping timestamp(datetime), symbol(str), open(float), high(float), low(float), close(float) and volume(float).

    Args:
        df (DataFrame): DataFrame to clean
    """
    # Clean dataset
    del df['ignore']
    del df['close_time']
    del df['quote_av']
    del df['trades']
    del df['tb_base_av']
    del df['tb_quote_av']

    # Add symbol column
    df.insert(1, 'symbol', str(symbol))

    # Convert to numeric
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    df['volume'] = pd.to_numeric(df['volume'])

    # Convert time (ms) to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

#DF cleaning
clean_data(df)

#------------------------------------------------------------------------------------------------
# LOAD DATA IN SQL DATABASE

# Database settings
username = 'user'
password = 'secretpw'
hostname = 'localhost'
port = '3306'
database = 'ProjetOPA'
table = 'data_market'

db_url = f'mysql://{username}:{password}@{hostname}:{port}/{database}'

# Create engine
engine = create_engine(db_url)

df.to_sql(name=table, con=engine, if_exists='replace', index=False)

try:
    connection = engine.connect()
    print("Connection successful!")
    connection.close()
except Exception as e:
    print("Connection failed:", e)


def check_existing_data(engine, table, symbol, timestamp):
    """Check if data already exists in the specified table for a given symbol and timestamp."""
    with engine.connect() as conn:
        # Construire la requête SQL pour vérifier si les données existent déjà
        query = f"SELECT COUNT(*) FROM {table} WHERE symbol = '{symbol}' AND timestamp = '{timestamp}'"
        
        # Exécuter la requête SQL
        result = conn.execute(query)
        
        # Récupérer le résultat
        count = result.fetchone()[0]
        
        # Si le nombre de lignes retourné est supérieur à zéro, les données existent déjà
        return count > 0




# Load data in database
'''
def load_data(df, db_url, table):
    """Load the DataFrame DF in the table, if each row of the DF doesn't already exists in the table

    Args:
        df (DataFrame): DataFrame to load to the table
        db_url (str): url of the database
        table (str): table name
    """
    for index in range(len(df)):
        row = df.iloc[index]
        if check_existing_data(engine, table, row['symbol'], row['timestamp']): 
            print(f"Data already exists in {table} for symbol {row['symbol']} and timestamp {row['timestamp']}.")
        else:     
            # Extract the row as a DF
            row_df = pd.DataFrame([row])
            # Load the row in the table
            row_df.to_sql(name=table, con=engine, if_exists='append', index=False)
            print(f"Data for symbol {row['symbol']} and timestamp {row['timestamp']} successfully loaded into {table}.")
'''

#load_data(df, db_url, table)