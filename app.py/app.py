from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model and class labels
model = load_model('blood_cell_model.h5')  # Make sure the file name is correct
labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

app = Flask(__name__)

@app.route('/')
def front():
    return render_template('front.html')

@app.route('/home')
def home():
    return render_template('home.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Image not uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded image to static folder
    image_path = os.path.join('static', file.filename)
    file.save(image_path)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = labels[class_index]

    # Render result page with prediction and image path
    return render_template('result.html', prediction=class_name, image_path='/' + image_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
