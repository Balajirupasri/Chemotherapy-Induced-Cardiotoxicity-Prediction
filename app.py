from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import mysql.connector
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ACCURACY_VALUES = {
    'CNN + Transformer': 1.0,
    'CNN + LSTM': 0.90,
    'CNN + GRU': 0.89
}
# Load the trained model
model = load_model('cardiotoxicity_cnn_transformer_smote_model.h5')

# Clinical features
CLINICAL_FEATURES = [
    'heart_rate','age','weight','height','time','heart_rhythm','LVEF','PWT','LAd',
    'LVDd','LVSd','AC','antiHER2','HTA','DL','DM','smoker','exsmoker','ACprev',
    'antiHER2prev','RTprev','CIprev','ICMprev','ARRprev','VALVprev','cxvalv'
]

# Initialize scaler
scaler = StandardScaler()

# MySQL DB config (using updated database and table)
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    port=3306,
    password="",
    database="cardiotoxicity_system2"
)
cursor = conn.cursor()

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        cursor.execute("SELECT * FROM users1 WHERE username=%s", (username,))
        if cursor.fetchone():
            flash('Username already exists')
            return redirect(url_for('register'))

        cursor.execute("INSERT INTO users1 (name, email, username, password) VALUES (%s, %s, %s, %s)",
                       (name, email, username, password))
        conn.commit()
        flash('Registered successfully! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT * FROM users1 WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        if user:
            session['username'] = username
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('dataset')
        if file is None or file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        table_html = df.head().to_html(classes='table table-bordered table-hover', index=False)
        return render_template('upload.html', table_html=table_html)
    return render_template('upload.html')

@app.route('/algo')
def algo():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('algo.html', accuracies=ACCURACY_VALUES)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            inputs = [float(request.form[f].replace(',', '.')) for f in CLINICAL_FEATURES]
            clinical_data = scaler.fit_transform([inputs])

            image_file = request.files['image']
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128)) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict([img, clinical_data])
            label = np.argmax(prediction)
            confidence = np.max(prediction)

            result = "Cardiotoxicity (CTRCD)" if label == 1 else "No Cardiotoxicity (NO_CTRCD)"
            return render_template('predict.html', prediction_text=f"Result: {result} with confidence {confidence:.2%}")

        except Exception as e:
            return render_template('predict.html', prediction_text=f"Error: {str(e)}")

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
