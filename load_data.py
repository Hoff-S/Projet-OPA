import pandas as pd
from sqlalchemy import create_engine
from make_DF import df

username = 'user'
password = 'secretpw'
hostname = 'localhost'
port = '3306'
database = 'ProjetOPA'

db_url = f'mysql://{username}:{password}@{hostname}:{port}/{database}'

engine = create_engine(db_url)

df.to_sql(name='data_market', con=engine, if_exists='replace', index=True)