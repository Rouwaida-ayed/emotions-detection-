from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model(r'C:\Users\rouwa\eclipse-workspace\CNN Image classification\HappyOSad.h5')

# Define the image dimensions expected by the model
img_height, img_width = 48, 48

# Define a route to serve the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Function to perform face detection
def detect_faces(image):
    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

# Define a route to handle image upload and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']
        
        # Read the image using OpenCV
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Perform face detection
        faces = detect_faces(image)
        
        # Check if any faces are detected
        if len(faces) == 0:
            # If no faces are detected, return an error message
            return render_template('error.html', message='No face detected in the uploaded image')
        
        # Preprocess the image
        resized_image = cv2.resize(image, (img_height, img_width))
        normalized_image = resized_image / 255.0
        input_image = np.expand_dims(normalized_image, axis=0)
        
        # Use the model to make predictions
        prediction = model.predict(input_image)
        
        # Interpret the prediction result
        if prediction[0] < 0.5:
            result = 'sad'
        else:
            result = 'happy'
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
