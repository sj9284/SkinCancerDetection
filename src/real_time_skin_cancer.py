import sys
import os
import numpy as np
import cv2
import hashlib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QLineEdit,
                             QPushButton, QHBoxLayout, QMessageBox, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from tensorflow.keras.models import load_model
import tensorflow as tf

# Define paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')
DATA_DIR = os.path.join(BASE_DIR, '../data')
SIZE = 32
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CONFIDENCE_THRESHOLD = 70.0
USE_KERAS = True  # Toggle between Keras and TFLite
USERS_FILE = os.path.join(DATA_DIR, 'users.txt')

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
        with open(USERS_FILE, 'a') as f:
            f.write(f"{username}:{hash_password(password)}\n")
        return True
    return False

# Preprocess image for prediction
def preprocess_image(frame):
    img = cv2.resize(frame, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict class from frame
def predict(frame):
    img_array = preprocess_image(frame)
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

# GUI Classes
class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login")
        self.setGeometry(300, 300, 300, 200)

        layout = QVBoxLayout()
        self.username_input = QLineEdit(self)
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit(self)
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.login_button = QPushButton("Login", self)
        self.login_button.clicked.connect(self.login)
        self.register_button = QPushButton("Register", self)
        self.register_button.clicked.connect(self.register)

        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)
        layout.addWidget(self.register_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        users = load_users()
        if username in users and users[username] == hash_password(password):
            self.hide()
            self.main_window = SkinCancerWindow(username, self)
            self.main_window.show()
        else:
            QMessageBox.warning(self, "Error", "Invalid username or password")

    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if save_user(username, password):
            QMessageBox.information(self, "Success", "Registration successful!")
        else:
            QMessageBox.warning(self, "Error", "Username already exists")

class SkinCancerWindow(QMainWindow):
    def __init__(self, username, login_window):
        super().__init__()
        self.username = username
        self.login_window = login_window
        self.setWindowTitle(f"Real-Time Skin Cancer Detection - {username}")
        self.setGeometry(100, 100, 800, 600)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            QMessageBox.critical(self, "Error", "Could not open webcam.")
            sys.exit(1)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        self.result_label = QLabel("Prediction: None", self)
        self.result_label.setStyleSheet("font-size: 16px;")

        self.history_text = QTextEdit(self)
        self.history_text.setReadOnly(True)
        self.history_text.setFixedHeight(80)

        self.start_button = QPushButton("Start Detection", self)
        self.start_button.clicked.connect(self.toggle_detection)
        self.logout_button = QPushButton("Logout", self)
        self.logout_button.clicked.connect(self.logout)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.history_text)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.logout_button)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.detection_active = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def toggle_detection(self):
        self.detection_active = not self.detection_active
        self.start_button.setText("Stop Detection" if self.detection_active else "Start Detection")
        if self.detection_active:
            self.timer.start(30)
        else:
            self.timer.stop()
            self.result_label.setText("Prediction: None")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.detection_active:
                pred_class, confidence = predict(frame_rgb)
                text = f"{pred_class} ({confidence:.2f}%)"
                color = (0, 255, 0) if pred_class != "Uncertain" else (255, 0, 0)
                cv2.putText(frame_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                self.result_label.setText(f"Prediction: {text}")
                self.history_text.append(f"{self.username} - {text}")
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def logout(self):
        self.timer.stop()
        self.cap.release()
        self.close()
        self.login_window.show()

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())
