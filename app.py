from flask import Flask, render_template, request
import numpy as np
import cv2 as cv
#from tensorflow.keras import models
from tensorflow.keras.models import load_model 

app = Flask(__name__)

# Load the pre-trained model
#model = models.load_model('image_classifier.model')
model = load_model('model/image_classifier.keras')

# Class names for CIFAR-10
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  
#changes made to this file to trigger pre commit.
# adding this comment to test the newly added workflow (run_test.yml)
def preprocess_image(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (32, 32))
    img = img / 255.0
    img = img[None, :]
    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the uploaded image
        file = request.files['file']
        if file:
            # Save the uploaded image
            img_path = 'uploads/uploaded_image.png'
            file.save(img_path)

            # Preprocess the image
            img = preprocess_image(img_path)

            # Make a prediction
            prediction = model.predict(img)
            index = np.argmax(prediction)
            result = class_names[index]

            return render_template('index.html', result=result, image_path=img_path)

    return render_template('index.html', result=None, image_path=None)


if __name__ == '__main__':
    app.run(debug=True)
