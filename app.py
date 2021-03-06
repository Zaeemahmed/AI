import os
from flask import Flask, flash, request, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from PIL import ImageFile, Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE=160


UPLOAD_FOLDER = '/static/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg'])
model = tf.keras.models.load_model('colorizer.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return "Welcome"

@app.route('/upload', methods=['POST'])
def fileUpload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    img = (Image.open(file))
    img = np.asarray(img)
    Image.fromarray(img).save('./test.png')
    img = cv2.imread('./test.png',1)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.clip(model.predict(img.reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img)
    img.save('./test.png')
    return send_file('test.png', mimetype='image/png', as_attachment=True)



if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run()

CORS(app, expose_headers='Authorization')