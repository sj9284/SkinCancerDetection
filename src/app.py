import os
import sys
import numpy as np
import cv2
import hashlib
import base64
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from tensorflow.keras.models import load_model
import tensorflow as tf
# Define paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')
DATA_DIR = os.path.join(BASE_DIR, '../data')
STATIC_DIR = os.path.join(BASE_DIR, '../static')
TEMPLATE_DIR = os.path.join(BASE_DIR, '../templates')
SIZE = 32
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CONFIDENCE_THRESHOLD = 70.0
USE_KERAS = True  # Toggle between Keras and TFLite
USERS_FILE = os.path.join(DATA_DIR, 'users.txt')

# Create Flask app
app = Flask(__name__, 
            static_folder=STATIC_DIR,
            template_folder=TEMPLATE_DIR)
app.secret_key = 'skin_cancer_detection_secret_key'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load the model
if USE_KERAS:
    model_path = os.path.join(MODEL_DIR, 'my_model.h5')
    if not os.path.exists(model_path):
        print(f"Error: Keras model not found at {model_path}")
        sys.exit(1)
    model = load_model(model_path)
    print("Keras model loaded successfully.")
    interpreter = None
else:
    model_path = os.path.join(MODEL_DIR, 'model.tflite')
    if not os.path.exists(model_path):
        print(f"Error: TFLite model not found at {model_path}")
        sys.exit(1)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")
    model = None

# User management functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            for line in f:
                if ':' in line:
                    username, hashed_pass = line.strip().split(':', 1)
                    users[username] = hashed_pass
    return users

def save_user(username, password):
    users = load_users()
    if username not in users:
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        with open(USERS_FILE, 'a') as f:
            f.write(f"{username}:{hash_password(password)}\n")
        return True
    return False

# Preprocess image for prediction
def preprocess_image(img_data):
    # Decode base64 image
    img_data = img_data.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Preprocess
    img = cv2.resize(frame, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict class from frame
def predict(img_data):
    img_array = preprocess_image(img_data)
    if USE_KERAS:
        predictions = model.predict(img_array, verbose=0)
    else:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    predicted_class = CLASSES[class_idx] if confidence >= CONFIDENCE_THRESHOLD else "Uncertain"
    return predicted_class, confidence

# Routes
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('detection'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    users = load_users()
    
    if username in users and users[username] == hash_password(password):
        session['username'] = username
        return redirect(url_for('detection'))
    return render_template('login.html', error="Invalid username or password")

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if save_user(username, password):
        return render_template('login.html', message="Registration successful!")
    return render_template('login.html', error="Username already exists")

@app.route('/detection')
def detection():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('detection.html', username=session['username'])

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    img_data = request.json.get('image')
    if not img_data:
        return jsonify({'error': 'No image data'}), 400
    
    try:
        predicted_class, confidence = predict(img_data)
        return jsonify({
            'class': predicted_class,
            'confidence': float(confidence),
            'username': session['username']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
