#!/bin/bash
cd "/home/christophe/Documents/Professionnel/Formation Datascientest DE/Projets/Projet-OPA"
python3 get_klines.py "01 January 2022"
python3 load_histo_klines_mongo.py

