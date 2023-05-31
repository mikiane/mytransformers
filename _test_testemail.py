import __lib_transformers
import os
from dotenv import load_dotenv
load_dotenv('.env')
from __lib_transformers import searchembedding
APP_PATH = os.environ['APP_PATH']

filename = APP_PATH + "datas/_contact@mikiane.com_outputsummary_20230521124159-29.txt"
__lib_transformers.mailfile(filename,"michel@brightness.fr")

