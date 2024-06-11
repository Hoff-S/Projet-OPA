from fastapi.testclient import TestClient

from fastapi_main import api

client = TestClient(api)

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {'message':'Ready'}

def test_get_last():
    response = client.get("/last/BTCEUR")
    assert response.status_code == 200

def test_get_all():
    response = client.get("/all/BTCEUR")
    assert response.status_code == 200

def test_post_insert():
    response = client.post("/insert", json={"open_time": "2020-01-01T00:00:00",\
                                           "symbol": "test",\
                                           "open_price": 0,\
                                           "high_price": 0,\
                                           "low_price": 0,\
                                           "close_price": 0,\
                                           "volume": 0})
    assert response.status_code == 200
    assert response.json() == {'message':'Added'}

def test_get_statistics():
    response = client.get("/statistics")
    assert response.status_code == 200

