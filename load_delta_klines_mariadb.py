import pandas as pd
from binance.client import Client
from sqlalchemy import create_engine
from datetime import datetime
from historical_data import clean_data

#------------------------------------------------------------------------------------------------

# Define which symbol data market constantly add to the database
symbol = 'BTCEUR'

# Define Client and download data
klinesT = Client().get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR)
df = pd.DataFrame(klinesT, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

# Only keeping the last 24h data
df = df.tail(24)

# DF cleaning
clean_data(df)

#Test
#print(df)

# Check if data already exists in database before loading it
def check_existing_data(engine, symbol, date):
    """Check if data already exists in the database."""
    query = f"SELECT COUNT(*) FROM data_market WHERE symbol = '{symbol}' AND timestamp >= '{date}'"
    result = engine.execute(query)
    count = result.fetchone()[0]
    return count > 0