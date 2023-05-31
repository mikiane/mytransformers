# -*- coding: utf-8 -*-
'''
Filename: transformers_api.py
Author: Michel Levy Provencal
Description: This file serves as the server for handling various API requests. It includes routes for summarizing documents, transforming text, and generating podcasts from text.
'''

from flask import Flask, request, jsonify, Response  # Import necessary modules from flask
from flask_cors import CORS  # Import CORS from flask_cors
# Import necessary custom library modules
from __lib_transformers import searchembedding
import __lib_transformers


app = Flask(__name__)  # Initialize Flask app
CORS(app)  # Enable CORS for the Flask app
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limit the size of uploaded files to 50 MB




# Function that implements a summarize API call
@app.route('/sumup', methods=['POST'])  # Define route for summarizing files
def sumup():
    if 'file' not in request.files:  # If no file is included in request
        return jsonify({'error': 'No file provided'}), 400
    uploaded_file = request.files['file']  # Get the uploaded file
    if uploaded_file.filename == '':  # If the file name is empty
        return jsonify({'error': 'No file provided'}), 400
    email = request.form.get('email')  # Get email from form data
    n = request.form.get('facteur')  # Get factor from form data
    model = request.form.get('model', "gpt-4")  # Get model from form data

    # Read the uploaded file
    inputstring = uploaded_file.read().decode('utf-8')
    # Summarize the text and get the filename of the summary
    filename = __lib_transformers.summarizelarge_chap(inputstring, str(email), n, model)
    # Send the summarized file to the specified email
    __lib_transformers.mailfile(filename, email, uploaded_file.filename)
    res = [{'id':1,'request':'summarize','answer':filename}]
    response = jsonify(res)
    response.headers['Content-Type'] = 'application/json'
    return(response)



# Function that implements a transformation (instruction) API call
@app.route('/transform', methods=['POST'])  # Define route for transforming text
def transform():
    email = request.form.get('email')  # Get email from form data
    text = request.form.get('text')  # Get text from form data
    instruction = request.form.get('instruction')  # Get instruction from form data
    model = request.form.get('model', "gpt-4")  # Get model from form data

    # Transform the text and get the filename of the transformed text
    filename = __lib_transformers.transform_chap(text, str(email), instruction, 1, model)
    # Send the transformed file to the specified email
    __lib_transformers.mailfile(filename, email, str(instruction))
    res = [{'id':1,'request':'transform','answer':filename}]
    response = jsonify(res)
    response.headers['Content-Type'] = 'application/json'
    return(response)


# Function that generates a podcast based on a text and send the file to an email adress
@app.route('/podcast', methods=['POST'])  # Define route for generating podcasts
def podcast():
    text = request.form.get('text')  # Get text from form data
    email = request.form.get('email')  # Get email from form data

    # Generate podcast from the text and get the filename of the podcast
    filename = __lib_transformers.synthesize_multi(text)
    # Send the podcast file to the specified email
    __lib_transformers.mailfile(filename, email)
    res = [{'id':1,'request':text,'answer':filename}]
    response = jsonify(res)
    response.headers['Content-Type']
