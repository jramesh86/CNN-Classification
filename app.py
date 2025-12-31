from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'model.h5'

# Custom wrapper to ignore quantization_config
def Dense_no_quantization(*args, **kwargs):
    if 'quantization_config' in kwargs:
        kwargs.pop('quantization_config')
    return Dense(*args, **kwargs)

# Load model safely once
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False, custom_objects={'Dense': Dense_no_quantization})
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

# Image preprocessing function
def preprocess_image(image, target_size=(100, 100)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # same as reshape
    return arr

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            return render_template('index.html', result='Error: Model not loaded'), 500

        if 'image' not in request.files:
            return redirect(url_for('index'))

        file = request.files['image']
        if file.filename == '':
            return redirect(url_for('index'))

        img = Image.open(file.stream)
        x = preprocess_image(img, target_size=(100, 100))
        
        proba = model.predict(x)[0][0]
        confidence_margin = abs(proba - 0.5)
        
        if confidence_margin > 0.3:
            label = 'cat' if proba > 0.5 else 'dog'
        else:
            label = 'neither'
        
        # Save uploaded image safely
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.stream.seek(0)
        with open(save_path, 'wb') as f:
            f.write(file.stream.read())
        
        return render_template('index.html', result=label, prob=float(proba), img_path=save_path)
    
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return render_template('index.html', result=f'Error: {str(e)}'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
