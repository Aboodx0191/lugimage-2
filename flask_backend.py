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
    "password": "Abood",  
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


def isValidXRAY(image):
    """Improved X-ray validation with better error handling"""
    try:
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Get histogram of pixel intensities
        hist = image.histogram()
        
        # Safety check for empty/unusual images
        if not hist or len(hist) < 256:
            return False
            
        total_pixels = sum(hist)
        if total_pixels == 0:  # Avoid division by zero
            return False
            
        # Calculate percentage of pixels in typical X-ray range
        # (X-rays usually have most pixels in mid-range intensities)
        mid_range = sum(hist[50:200]) 
        ratio = mid_range / total_pixels
        
        # Adjust threshold as needed (0.6-0.8 works well for most X-rays)
        return ratio >= 0.7
    except Exception as e:
        print(f"X-ray validation error: {str(e)}")
        return False


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
        # Create uploads directory if needed
        upload_folder = "uploads"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        # Save the file temporarily
        image_path = os.path.join(upload_folder, file.filename)
        
        # First save the file to disk
        file.save(image_path)
        
        # Open and validate the image
        image = Image.open(image_path)
        
        # Reset file pointer (important!)
        image = Image.open(image_path)
        
        if not isValidXRAY(image):
            os.remove(image_path)  # Clean up invalid image
            return jsonify({
                "error": "Invalid X-ray image",
                "message": "The uploaded image doesn't meet X-ray quality standards"
            }), 400

        # Process and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        confidence = float(prediction[0][0])
        prediction_label = "Pneumonia" if confidence > 0.5 else "Normal"
        confidence = confidence if prediction_label == "Pneumonia" else 1 - confidence

        # Save to database
        save_prediction_to_db(image_path, prediction_label, confidence)

        return jsonify({
            "prediction": prediction_label, 
            "confidence": confidence,
            "message": "Successfully analyzed X-ray image"
        })
        
    except Exception as e:
        # Clean up if something went wrong
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

# Serve index.html at the root URL
@app.route("/")
def index():
    return send_file("index.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)