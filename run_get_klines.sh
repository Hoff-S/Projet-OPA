#!/bin/bash
cd "/home/christophe/Documents/Professionnel/Formation Datascientest DE/Projets/Projet-OPA"
python3 get_klines.py
python3 load_delta_klines_mongo.py

