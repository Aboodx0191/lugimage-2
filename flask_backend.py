from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os


# Initialize Flask app
app = Flask(__name__)
CORS(app) 
@app.route("/")
def index():
    return send_file("index.html")


# Load the trained model
model = load_model("best_vgg16_model.h5")

# Preprocessing function for input images
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to 224x224
    image = np.array(image.convert("RGB")) / 255.0   
    image = np.expand_dims(image, axis=0)  
    return image

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    print("Request received")  # Debugging
    print(f"Request files: {request.files}")  # Debugging

    if "file" not in request.files:
        print("No file part in the request")  # Debugging
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]  # Get the uploaded file
    
    # Debugging: Check if the file is empty
    if file.filename == "":
        print("Empty file uploaded")  # Debugging
        return jsonify({"error": "Empty file uploaded"}), 400

    # Debugging: Check if the file is an image
    if not file.content_type.startswith("image/"):
        print(f"Uploaded file is not an image: {file.content_type}")  # Debugging
        return jsonify({"error": "Uploaded file is not an image"}), 400

    print(f"File received: {file.filename}, Content-Type: {file.content_type}")  # Debugging

    try:
        image = Image.open(io.BytesIO(file.read()))  # Open image
        processed_image = preprocess_image(image)  # Preprocess image

        prediction = model.predict(processed_image)  # Make prediction
        confidence = float(prediction[0][0]) 
        prediction = "Pneumonia"  if confidence > 0.5 else "Normal"  # Convert to label
        confidence = confidence if prediction =="Pneumonia" else 1 - confidence
        return jsonify({"prediction": prediction, "confidence" : confidence})  # Return response as JSON
    except Exception as e:
        print(f"Error processing image: {e}")  # Debugging
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)


