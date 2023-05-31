import __lib_transformers


import os
from dotenv import load_dotenv

load_dotenv('.env')

APP_PATH = os.environ['APP_PATH']

# Configurer les informations de connexion pour Amazon Polly et le serveur de messagerie
project_folder = APP_PATH 


#SUMMARY SIMPLE
# ouvrir un fichier txt et récupérer le contenu dans la variable inputstring
inputstring = open(APP_PATH + "datas/book.txt", "r").read()
email = "michel@brightness.fr"
instruction = "extraire les points clés"
filename = __lib_transformers.transform_chap(inputstring, str(email), instruction, 1)
__lib_transformers.mailfile(filename, email, str(instruction))


# Lancer la fonction summarize_and_send_email dans un thread séparé
#thread = threading.Thread(target=summarize_and_send_email, args=(inputstring, email))
#thread.start()
