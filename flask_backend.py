from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import mysql.connector
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
model = load_model("best_vgg16_model.h5")

# Database configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "Abood",  # Replace with your MySQL root password
    "database": "pneumonia_detection"
}

# Function to save prediction to the database
def save_prediction_to_db(image_path, prediction, confidence):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = "INSERT INTO predictions (image_path, prediction, confidence) VALUES (%s, %s, %s)"
        cursor.execute(query, (image_path, prediction, confidence))
        connection.commit()
        cursor.close()
        connection.close()
        print("Prediction saved to database")
    except Exception as e:
        print(f"Error saving prediction to database: {e}")

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to 224x224
    image = np.array(image.convert("RGB")) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file uploaded"}), 400

    if not file.content_type.startswith("image/"):
        return jsonify({"error": "Uploaded file is not an image"}), 400

    try:
        # Save the uploaded image to a folder
        upload_folder = "uploads"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        image_path = os.path.join(upload_folder, file.filename)
        file.save(image_path)

        # Open and preprocess the image
        image = Image.open(image_path)
        processed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(processed_image)
        confidence = float(prediction[0][0])
        prediction_label = "Pneumonia" if confidence > 0.5 else "Normal"
        confidence = confidence if prediction_label == "Pneumonia" else 1 - confidence

        # Save the prediction to the database
        save_prediction_to_db(image_path, prediction_label, confidence)

        # Return the prediction result
        return jsonify({"prediction": prediction_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve index.html at the root URL
@app.route("/")
def index():
    return send_file("index.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)