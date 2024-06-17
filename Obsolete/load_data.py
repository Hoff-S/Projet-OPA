import pandas as pd
from pymongo import MongoClient
from make_DF import df

client = MongoClient('localhost', 27017)
db = client['ProjetOPA']
collection = db['market_data']

# Convertir le DataFrame en une liste de dictionnaires
data = df.to_dict('records', index=True)

# Insérer les données dans la collection MongoDB
collection.insert_many(data)

# Fermer la connexion MongoDB
client.close()