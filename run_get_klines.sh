#!/bin/bash
cd "/home/christophe/Projets/Projet-OPA"
python3 get_klines.py
python3 load_delta_klines_mongo.py

