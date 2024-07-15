import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import mysql.connector
from mysql.connector import Error
import logging
import json
from PIL import Image 

# Membuat instance Flask
app = Flask(__name__)
CORS(app)

# Memuat model yang sudah disimpan
model_path = 'C:/xampp/htdocs/WEB-IS_USG/public/models/best_model.keras'
model = load_model(model_path)

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

# Koneksi ke database MySQL
def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='your_database',
            user='root',
            password=''
        )
        if connection.is_connected():
            return connection
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        return None

# Definisikan route untuk root
@app.route('/')
def home():
    return "API is running"

# Definisikan route untuk endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan file gambar dari request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400
        
        if file:
            # Membaca gambar dan mengubahnya menjadi array numpy
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((128, 128))
            input_data = np.array(img)

            if input_data.shape != (128, 128, 3):
                return jsonify({"error": "Invalid input shape, must be 128x128x3"}), 400
            
            input_data = input_data.reshape(1, 128, 128, 3)
            print(input_data)
            # Melakukan prediksi
            prediction = model.predict(input_data)

            predicted_class = np.argmax(prediction, axis=1)[0]
            hasil=['tidak hamil','hamil']
            
            # Simpan hasil ke database MySQL
            connection = create_db_connection()
            if connection is not None:
                cursor = connection.cursor()
                sql_query = "INSERT INTO predictions (predicted_class) VALUES (%s)"
                cursor.execute(sql_query, (hasil[predicted_class],))
                connection.commit()
                cursor.close()
                connection.close()
            else:
                return jsonify({"error": "Database connection failed"}), 500
            
            # Mengembalikan hasil prediksi sebagai JSON
            response = {
                'prediction': str(hasil[predicted_class])
            }
            return jsonify(response)
        else:
            return jsonify({"error": "File not found"}), 400
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 400
# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
