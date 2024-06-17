import pandas as pd
from pymongo import MongoClient
import os

username = "admin"
password = os.environ.get('MONGODB_PW')

client = MongoClient('mongodb://%s:%s@127.0.0.1:27017' % (username, password))
db = client['cryptobot']

col_name = "klines"
collection = db[col_name]

# Récupération des données fraiches
histo_file = "crypto_histo.json"
df = pd.read_json(histo_file, orient ='split', compression = 'infer')

# Pour chaque ligne dans le dataframe, tester l'existence dans mongo
for idx in range(len(df)):
  if collection.find_one({'open_time' : df['open_time'].iloc[idx], "symbol" : df['symbol'].iloc[idx]}):
    print(df['open_time'].iloc[idx], df['symbol'].iloc[idx], " existe déjà")
  else:
    print(df['open_time'].iloc[idx], df['symbol'].iloc[idx], " va être créé")
    collection.insert_one(df.iloc[[idx]].to_dict('records')[0])

# Fermer la connexion MongoDB
client.close()