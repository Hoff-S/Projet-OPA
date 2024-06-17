import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import sys

def prepare_file(date_toget):
    klinesT = Client().get_historical_klines("BTCEUR", Client.KLINE_INTERVAL_1HOUR, date_toget)
    df = pd.DataFrame(klinesT, columns = ['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

    # Clean dataset

    del df['ignore']
    del df['close_time']
    del df['quote_av']
    del df['trades']
    del df['tb_base_av']
    del df['tb_quote_av']

    df.insert(1,'symbol',"BTCEUR") 
    df['close_price'] = pd.to_numeric(df['close_price'])
    df['high_price'] = pd.to_numeric(df['high_price'])
    df['low_price'] = pd.to_numeric(df['low_price'])
    df['open_price'] = pd.to_numeric(df['open_price'])
    df['volume'] = pd.to_numeric(df['volume'])

    # Convert time (ms) to datetime

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

    # écriture du fichier json contenant l'historique
    histo_file = 'crypto_histo.json' 
    df.to_json(histo_file, orient = 'split', compression = 'infer', index = 'true')

    print(df.head())

date_format = "%d %B %Y"

if len(sys.argv) == 2:
    # test du bon format
    date_string=sys.argv[1]    
    try:
        date_toget = datetime.strptime(date_string, date_format)
        date_toget = date_string
        print("Récupération des données depuis {}".format(date_toget))
        prepare_file(date_toget)
    except ValueError:
        print("Format de date incorrect ex. : 01 January 2022")
elif len(sys.argv) == 1:
    print("Récupération des données depuis la veille")
    date_datetime = datetime.today() - timedelta(days=1)
    date_toget = date_datetime.strftime(date_format)
    prepare_file(date_toget)
else:
    print("Trop de paramètres")
    



