import requests as re
import os


def test_get_status(api_url):
    response = re.get(f"{api_url}/status")
    assert response.status_code == 200
    assert response.json() == {'message':'Ready'}

def test_get_last(api_url):
    response = re.get(f"{api_url}/last/BTCEUR")
    assert response.status_code == 200

def test_get_all(api_url):
    response = re.get(f"{api_url}/all/BTCEUR")
    assert response.status_code == 200

def test_post_insert(api_url):
    response = re.post(f"{api_url}/insert", json={"open_time": "2020-01-01T00:00:00",\
                                           "symbol": "test",\
                                           "open_price": 0,\
                                           "high_price": 0,\
                                           "low_price": 0,\
                                           "close_price": 0,\
                                           "volume": 0})
    assert response.status_code == 200
    assert response.json() == {'message':'Added'}

def test_get_statistics(api_url):
    response = re.get(f"{api_url}/statistics")
    assert response.status_code == 200

def test_get_prediction(api_url):
    response = re.get(f"{api_url}/prediction")
    assert response.status_code == 200

# Define url for api.
api_url = "http://{}:8000".format(os.environ.get('API_URL'))
# list of tests.
test_get_status(api_url)
test_get_last(api_url)
test_get_all(api_url)
test_post_insert(api_url)
test_get_statistics(api_url)
test_get_prediction(api_url)
