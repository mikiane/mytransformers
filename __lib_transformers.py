
'''
Filename: __lib_transformers.py
Author: Michel Levy Provencal
Description: This script includes a variety of functions designed Ffor text and audio transformation using OpenAI's GPT-3 API and Amazon Polly.
'''
# Import the necessary libraries
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import openai
import boto3
import os
import random
from datetime import datetime
import pydub
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64
import mimetypes
import time
import sys
import csv
import random
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
import time
import csv
import random
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv('.env')

# Environment Variables
SENDGRID_KEY = os.environ['SENDGRID_KEY']
APP_PATH = os.environ['APP_PATH']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
AWS_SECRET_KEY = os.environ['AWS_SECRET_KEY']
REGION_NAME = os.environ['REGION_NAME']

# instantiate an Amazon Polly client
polly = boto3.client('polly', region_name=REGION_NAME,
                     aws_access_key_id=AWS_ACCESS_KEY,
                     aws_secret_access_key=AWS_SECRET_KEY)

# function to break down input text into smaller segments and then use Polly to generate speech
def synthesize_multi(inputtext):

    # define a maximum number of characters for each Polly API call
    max_chars = 2500
    segments = []

    # break down the input text into sentences
    sentences = inputtext.split('. ')
    current_segment = ''

    # iterate over each sentence and add to the current segment until the limit is reached
    for sentence in sentences:
        if len(current_segment) + len(sentence) + 1 <= max_chars:
            current_segment += sentence + '. '
        else:
            segments.append(current_segment)
            current_segment = sentence + '. '

    # add the last segment if it is not empty
    if current_segment:  
        segments.append(current_segment)

    # set up an output directory and a list to store paths to output files
    output_dir = APP_PATH + 'datas/'
    output_files = []

    # iterate over each segment
    for i, segment in enumerate(segments):
        print("Segment number :" + str(i))
        print("\n" + segment)
        
        # get the current time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current time is :", current_time)

        # prepare the text for the Polly API and make the request
        ssml_segment = "<speak><prosody rate=\"90%\">" + str(segment) + "</prosody></speak>"
        response = polly.synthesize_speech(
            OutputFormat='mp3',
            VoiceId='Remi',
            TextType='ssml',
            Text=ssml_segment,
            LanguageCode='fr-FR',
            Engine='neural'
        )

        print("API response received")
        # get the current time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current time is :", current_time)
        audio_stream = response.get('AudioStream')
        audio_data = audio_stream.read()

        # generate a unique filename and save the audio data to a file
        filename = f"audiooutput_segment{i}.mp3"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(audio_data)

        # add the path to the output file to the list
        output_files.append(output_path)

    # concatenate all the audio files together
    combined_audio = pydub.AudioSegment.silent(duration=0)
    for output_file in output_files:
        segment_audio = pydub.AudioSegment.from_mp3(output_file)
        combined_audio += segment_audio

    # generate a filename for the final output file
    final_filename = "audiooutput" + str(random.randint(1, 10000)) + ".mp3"
    final_output_path = os.path.join(output_dir, final_filename)

    # save the combined audio to a file
    combined_audio.export(final_output_path, format='mp3')

    # return the path to the final output file
    return (output_dir + final_filename)



# Function to get the text embedding from OpenAI's API
def get_embedding(text, model="text-embedding-ada-002"):
    openai.api_key = OPENAI_API_KEY
    text = text.replace("\n", " ")  # Replaces newline characters with spaces
    return openai.Embedding.create(input = [text], engine=model)['data'][0]['embedding']  # Returns the embedding



# Function to search for a text within a local dataset using text embeddings
def searchembedding(text, filename):
    openai.api_key = OPENAI_API_KEY

    # Read the CSV file
    df = pd.read_csv(filename)

    # Convert the strings stored in the 'ada_embedding' column into vector objects
    df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

    # Convert the search term into a vector
    searchvector = get_embedding(text, model='text-embedding-ada-002')

    # Create a new column using cosine_similarity to compare the searchvector with each row
    df['similarities'] = df.ada_embedding.apply(lambda x: np.dot(x, searchvector))

    # Sort the rows by similarity and keep the most similar one
    res = df.sort_values('similarities', ascending=False).head(1)

    # Set pandas option to display all columns
    pd.set_option('display.max_columns', None)

    # Check if the 'combined' column exists in the DataFrame
    if 'combined' in res.columns:
        # Check if the DataFrame is not empty
        if not res.empty:
            # Check if the index is of integer type
            if res.index.dtype == 'int64':
                # Return all records
                return '\n'.join(res['combined'].values)
            else:
                return "L'index du DataFrame n'est pas de type entier"
        else:
            return "Le DataFrame est vide"
    else:
        return "La colonne 'combined' n'existe pas dans le DataFrame"



def mailfile(filename, destinataire, message=""):
    """
    Fonction pour envoyer un e-mail avec une pièce jointe via SendGrid.
    
    Args:
        filename (str): Le chemin vers le fichier à joindre.
        destinataire (str): L'adresse e-mail du destinataire.
        message (str, optional): Un message à inclure dans l'e-mail. Par défaut, le message est vide.
    """
    # Création de l'objet Mail
    message = Mail(
        from_email='contact@brightness.fr',
        to_emails=destinataire,
        subject='Le résultat du traitement' + message,
        plain_text_content='Votre demande a été traité, veuillez trouver votre fichier en pièce jointe. Fichier traité : ' + message)
    
    # Lecture du fichier à joindre
    with open(filename, 'rb') as f:
        data = f.read()

    # Encodage du fichier en base64
    encoded = base64.b64encode(data).decode()
    
    # Détermination du type MIME du fichier
    mime_type = mimetypes.guess_type(filename)[0]
    
    # Création de l'objet Attachment
    attachedFile = Attachment(
    FileContent(encoded),
    FileName(filename),
    FileType(mime_type),
    Disposition('attachment')
    )
    message.attachment = attachedFile

    # Tentative d'envoi de l'e-mail via SendGrid
    try:
        sg = SendGridAPIClient(SENDGRID_KEY)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)





# Function to split a large text into smaller blocks
def split_text_into_blocks(text, limit=4000):
    # Initialize variables
    blocks = []
    current_block = ""
    words = text.split()
    
    # Iterate over words
    for word in words:
        # Check if word fits in the current block
        if len(current_block + word) + 1 < limit:
            current_block += word + " "
        else:
            last_delimiter_index = max(current_block.rfind(". "), current_block.rfind("\n"))

            # Break block at the last complete sentence or newline
            if last_delimiter_index == -1:
                blocks.append(current_block.strip())
                current_block = word + " "
            else:
                delimiter = current_block[last_delimiter_index]
                blocks.append(current_block[:last_delimiter_index + (1 if delimiter == '.' else 0)].strip())
                current_block = current_block[last_delimiter_index + (2 if delimiter == '.' else 1):].strip() + " " + word + " "
    
    # Add the last block
    if current_block.strip():
        blocks.append(current_block.strip())

    return blocks

# Function to write blocks to a csv file
def write_blocks_to_csv(blocks, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for block in blocks:
            csvwriter.writerow([block])

# Function to generate a csv file from a string of text
def write_csv_from_string(text, filename):
    limit = 4000  # Limit for text blocks
    blocks = split_text_into_blocks(text, limit)  # Split text into blocks
    write_blocks_to_csv(blocks, filename)  # Write blocks to csv file

# Function to summarize text
def transform(text, instruct, model="gpt-4"):
    api_key = OPENAI_API_KEY
    if model=="gpt-4":
        limit = 12000  # Limit for text size
    else:
        limit = 6000
    prompt = instruct + "\n" + text[:limit] + ":\n"  # Construct the prompt
    system = "Je suis un assistant parlant parfaitement le français et l'anglais capable de corriger, rédiger, paraphraser, traduire, résumer, développer des textes."

    # Try to make a request to the API
    attempts = 0
    while attempts < 10:
        try:
            url = 'https://api.openai.com/v1/chat/completions'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            data = {
                'model': model,
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'system', 'content': system}
                ]
            }
            response = requests.post(url, headers=headers, json=data)
            json_data = response.json()
            message = json_data['choices'][0]['message']['content']
            return message.strip()
        except Exception as e:
            error_code = type(e).__name__
            error_reason = str(e)
            attempts += 1
            print(f"Erreur : {error_code} - {error_reason}. Nouvel essai dans 5 secondes...")
            time.sleep(5)

    print("Erreur : Echec de la création de la completion après 5 essais")
    sys.exit()

# Function to summarize a chapter of text
def transform_chap(text, prefix, instruct, n=3, model='gpt-4'):
    now = datetime.now()
    rand_str = str(now.strftime("%Y%m%d%H%M%S")) + "-"+ str(random.randint(0, 100))
    path = APP_PATH + "datas/"

    # Write input text to CSV
    input_f = path + "_" + prefix + "_input_" + rand_str +".csv"
    write_csv_from_string(text, input_f)

    # Summarize the text
    for j in range(1, n+1):
        # Update input filename
        if j > 1:
            input_f = output_f + "_" + str(j-1) + ".csv"

        with open(input_f, "r") as input_file:
            reader = csv.reader(input_file)
            # Update output filename
            output_f = path + "_" + prefix + "_output_" + rand_str
            with open(output_f + "_" + str(j) + ".csv", "w", newline="") as output_file:
                writer = csv.writer(output_file)
                rows_concatenated = []
                for row in reader:
                    rows_concatenated.append(row[0])
                    if (len(rows_concatenated) >= j) or (len(reader) == 0):
                        text = " ".join(rows_concatenated)
                        summary = transform(text, instruct, model)
                        writer.writerow([summary] + row[1:])
                        rows_concatenated = []

    # Write final summary to a text file
    outputxt = path + "_" + prefix + "_outputsummary_" + str(rand_str) + ".txt"
    with open(output_f + "_" + str(j) + ".csv", 'r') as csv_file, open(outputxt, 'w') as txt_file:
        csv_output = csv.reader(csv_file)
        for row in csv_output:
            txt_file.write(','.join(row) + '\n\n')

    return(outputxt)

# Function to split a large text into smaller blocks
def split_text_into_blocks(text, limit=4000):
    # Initialize variables
    blocks = []
    current_block = ""
    words = text.split()
    
    # Iterate over words
    for word in words:
        # Check if word fits in the current block
        if len(current_block + word) + 1 < limit:
            current_block += word + " "
        else:
            last_delimiter_index = max(current_block.rfind(". "), current_block.rfind("\n"))

            # Break block at the last complete sentence or newline
            if last_delimiter_index == -1:
                blocks.append(current_block.strip())
                current_block = word + " "
            else:
                delimiter = current_block[last_delimiter_index]
                blocks.append(current_block[:last_delimiter_index + (1 if delimiter == '.' else 0)].strip())
                current_block = current_block[last_delimiter_index + (2 if delimiter == '.' else 1):].strip() + " " + word + " "
    
    # Add the last block
    if current_block.strip():
        blocks.append(current_block.strip())

    return blocks

# Function to write blocks to a csv file
def write_blocks_to_csv(blocks, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for block in blocks:
            csvwriter.writerow([block])

# Function to generate a csv file from a string of text
def write_csv_from_string(text, filename):
    limit = 2000  # Limit for text blocks
    blocks = split_text_into_blocks(text, limit)  # Split text into blocks
    write_blocks_to_csv(blocks, filename)  # Write blocks to csv file

# Function to summarize text
def summarize(text, model='gpt-3.5-turbo'):
    api_key = OPENAI_API_KEY
    if model=="gpt-4":
        limit = 12000  # Limit for text size
    else:
        limit = 6000
    prompt = "Texte : " + text[:limit] + "\nTache : Résumer le texte en respectant le style et le sens. \
        \nFormat : Un texte court dont le style et le sens sont conformes au texte original. \
        \nObjectif : Obtenir un résumé sans introduction particulière. \
        \nEtapes : Ne jamais mentionner que le texte produit est un résumé. \
        \n Le résumé : \
        \n"
    system = "Rôle : Etre un rédacteur en français spécialisé dans le résumé d’ouvrages."

    # Try to make a request to the API
    attempts = 0
    while attempts < 100000:
        try:
            url = 'https://api.openai.com/v1/chat/completions'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            data = {
                'model': model,
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'system', 'content': system}
                ]
            }

            response = requests.post(url, headers=headers, json=data)
            json_data = response.json()
            message = json_data['choices'][0]['message']['content']
            return message.strip()
        except Exception as e:
            error_code = type(e).__name__
            error_reason = str(e)
            attempts += 1
            print(f"Erreur : {error_code} - {error_reason}. Nouvel essai dans 8 secondes...")
            time.sleep(1.1*attempts)

    print("Erreur : Echec de la création de la completion après x essais")

# Function to summarize large chapter
def summarizelarge_chap(text, prefix, n=3, model="gpt-4"):
    now = datetime.now()
    rand_str = str(now.strftime("%Y%m%d%H%M%S")) + "-"+ str(random.randint(0, 100))
    path = APP_PATH + "datas/"
    input_f = path + "_" + prefix + "_input_" + rand_str +".csv"
    output_f = path + "_" + prefix + "_output_" + rand_str

    # Write input to csv
    write_csv_from_string(text, input_f)
    j = 1

    # Summarize the text
    while j <= int(n):
        if j > 1:
            input_f = output_f + "_" + str(j-1) + ".csv"

        with open(input_f, "r") as input_file_count:
            reader = csv.reader(input_file_count)
            lines = sum(1 for _ in reader)

            if lines < j:
                break

        with open(input_f, "r") as input_file:
            reader = csv.reader(input_file)
            with open(output_f + "_" + str(j) + ".csv", "w", newline="") as output_file:
                writer = csv.writer(output_file)
                rows_concatenated = []
                for row in reader:
                    lines -= 1
                    rows_concatenated.append(row[0])

                    if (len(rows_concatenated) >= j) or (lines==0):
                        text = " ".join(rows_concatenated)
                        summary = summarize(text, model)
                        writer.writerow([summary] + row[1:])
                        rows_concatenated = []
            j += 1

    # Write final summary to a text file
    outputxt = path + "_" + prefix + "_outputsummary_" + str(rand_str) + ".txt"
    inputcsv = output_f + "_" + str(j-1) + ".csv"
    with open(inputcsv, 'r') as csv_file, open(outputxt, 'w') as txt_file:
        csv_output = csv.reader(csv_file)
        for row in csv_output:
            txt_file.write(','.join(row) + '\n\n')

    return(outputxt)



