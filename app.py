from flask import Flask, request, jsonify
from model import *
from flask_cors import CORS
global model_VAE
import numpy as np
import os
app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route("/predict", methods=["POST"])
def predict():
    input = np.zeros(n_items).reshape(1,-1)
    if request.get_json():
        data = request.get_json()
        for i in data['data']:
            input[0][i] = 1
        input = input.astype('float32')
        X = torch.FloatTensor(input)
        model_VAE.eval()
        logits_vad, KL = model_VAE(X)
        logits_vad, idx = torch.sort(logits_vad, descending= True)
        print(logits_vad[0][0:50])
        print(idx[0][0:40].tolist())
        result = {"data":idx[0][0:40].tolist() }
    return jsonify({"result":result})
if __name__ == '__main__':

    app.run(debug=True)
