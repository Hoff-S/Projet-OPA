import pandas as pd
from pymongo import MongoClient
import os

username = "admin"
password = os.environ.get('MONGODB_PW')

client = MongoClient('mongodb://%s:%s@127.0.0.1:27017' % (username, password))
db = client['cryptobot']

col_name = "klines"
histo_file = "crypto_histo.json"
# suppression de la collection si elle existe

collist = db.list_collection_names()
if col_name in collist:
  print("La collection {} existe et sera supprimée".format(col_name))
  collection = db[col_name]
  collection.drop()
else:
  print("La collection {} est créée".format(col_name))
  collection = db[col_name]

if os.path.exists(histo_file):
    # lecture du fichier json contenant l'historique
    df = pd.read_json(histo_file, orient ='split', compression = 'infer')

    data = df.to_dict('records')

    # Insérer les données dans la collection MongoDB
    histo_file_size = len(df)
    collection.insert_many(data)

    print("Les {} lignes du fichier {} ont été insérées dans la collection {}".format(histo_file_size,histo_file, col_name))
else:
    print("Le fichier {} n'existe pas.".format(histo_file))

# Fermer la connexion MongoDB
client.close()