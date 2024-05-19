from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
from main import class_names

model_name = "sportsPosesClassifier.keras"

model = load_model(model_name)
model.summary()


app = Flask(__name__)


@app.route('/api', methods=['GET'])
def home():
    return "Hello, welcome to sportoMov's API."


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']
    image = cv.resize(image, (125, 125))

    result = model.predict(np.array([image])/255)
    index = np.argmax(result)
    return jsonify(class_names[index])


if __name__ == '__app__':
    app.run(host='0.0.0.0', port=5000)
