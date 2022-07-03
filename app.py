from flask import Flask, request, jsonify
from model import *
from flask_cors import CORS
global model_VAE
import numpy as np
import os
model_VAE = None
app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    state_dict = torch.load('model_VAE_final.pth', map_location=torch.device('cpu'))
    encoder_dims = [n_items, 600, 200]
    decoder_dims = [200, 600, n_items]
    model_VAE = MultiVAE(encoder_dims=encoder_dims, decoder_dims=decoder_dims)
    model_VAE.load_state_dict(state_dict)
    app.run(debug=True)
