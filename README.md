
# Text Transformers API

This repository hosts a Flask API that provides various services related to text transformation, such as summarization, text transformation with custom instructions, and generating a podcast from text.

## Setup

To run the API, you first need to install the required Python packages. This can be done using pip:

pip install flask flask_cors
You also need to have the custom libraries __lib_transformers and searchembedding in your Python path.

## Usage
Start the server with:

python transformers_api.py
This will start a Flask server that listens for requests on localhost:5000.

### Summarize API
To summarize a text, send a POST request to localhost:5000/sumup. The request should include:

A file to be summarized
An email address to send the summarized document to
The summarization factor (optional)
The model to be used for summarization (optional, defaults to 'gpt-4')


### Transform API
To transform a text according to custom instructions, send a POST request to localhost:5000/transform. The request should include:

The text to be transformed
The transformation instruction
An email address to send the transformed text to
The model to be used for the transformation (optional, defaults to 'gpt-4')

### Podcast API
To generate a podcast from a text, send a POST request to localhost:5000/podcast. The request should include:

The text to be transformed into a podcast
An email address to send the podcast to
Limitations
The server is set to limit the size of uploaded files to 50 MB.

### Author
Michel Levy Provencal

You can, of course, modify this to better fit your needs ;-)