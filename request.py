import requests
from binance.spot import Spot

crypto_list = ['ETH', 'SOL', 'SHIB']

def create_data(crypto):
    """Fonction prenant en entrée une cryptomonnaie et renvoie un tuple : ('nom', prix en EUR)

    Args:
        crypto (String): Code diminutif de la cryptomonnaie (Exemple : Bitcoin = BTC)

    Returns:
        Tuple: Tuple dont le premier élément est le code diminutif de la cryptomonnaie 
        et dont le deuxième élément est son prix moyen en EURO sur les 5 dernières minutes.
    """
    prix = requests.get(f'https://testnet.binance.vision/api/v3/avgPrice?symbol={crypto}EUR').json()['price']
    return (str(crypto), prix)

def make_dict(list):
    """Fonction prenant en entrée une liste de cryptomonnaies et en renvoie un dictionnaire de leur prix moyen sur 5 minutes en EURO

    Args:
        list (list): Liste de diminutifs de cryptomonnaies (Exemple : BTC = Bitcoin)
    """
    dict = {}
    for crypto in list:
        dict[str(create_data(crypto)[0])] = create_data(crypto)[1]
    return dict

print(make_dict(crypto_list))