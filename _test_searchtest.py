import os
from dotenv import load_dotenv
load_dotenv('.env')

from __lib_transformers import searchembedding
APP_PATH = os.environ['APP_PATH']

text="eau rare"
#text = request.form.post('text')
resultat = searchembedding(text, APP_PATH + 'datas/33questions-embedding.csv')
print(resultat)