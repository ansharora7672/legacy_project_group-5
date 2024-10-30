#code taken from - https://flask.palletsprojects.com/en/stable/testing/
# https://gist.github.com/justinmklam/c453ea62b47f4e302dbd83c05882e3bd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pytest
from app import app
from result import model
import cv2 as cv

import io
from PIL import Image

import numpy as np
from tensorflow.keras import models

@pytest.fixture()
def client():
        app.config.update({
        "TESTING": True,
    })
        with app.test_client() as client:

            yield client


def test_request(client):
    response = client.get("/")
    assert b"<title>Image Classifier</title>" in response.data



def test_post_without_file(client):
    
    response = client.post("/", data={})
    assert response.status_code == 400
    assert b"Bad Request" in response.data

def test_valid_image_upload(client):
    img = Image.new("RGB", (32, 32), color="blue") 
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0) 

    data = {
        'file': (img_io, 'test_image.png')
    }
    response = client.post("/", data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"prediction" in response.data or any(cls.encode() in response.data for cls in ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])



def test_training_step():
    model = models.load_model("model/image_classifier.keras")
    # Create a batch of random data
    random_images = np.random.rand(4, 32, 32, 3).astype("float32")
    random_labels = np.random.randint(0, 10, 4)
    
    # Run one training step
    history = model.fit(random_images, random_labels, epochs=1, verbose=0)
    
    # Verify that the training loss is recorded
    if "loss" not in history.history:
        return "Failed: Loss should be recorded in the training history."
    
    return "Passed: Training step completed successfully."

import numpy as np
from tensorflow.keras import models

def test_prediction_shape():
    model = models.load_model("model/image_classifier.keras")
    # Create a batch of random data
    random_images = np.random.rand(5, 32, 32, 3).astype("float32")
    
    # Make predictions
    predictions = model.predict(random_images)
    
    # Check that the output shape is (5, 10)
    if predictions.shape != (5, 10):
        return "Failed: Predictions should have shape (5, 10)."
    
    return "Passed: Prediction shape is correct."

# This test checks if the model can make predictions for a larger batch of 10 images. It also ensures that the output shape is correct (10 images, each with predictions for 10 classes).
def test_model_accuracy(client):
    model = models.load_model("model/image_classifier.keras")
    
    # Create a batch of random test images
    test_images = [np.random.rand(32, 32, 3).astype("float32") for _ in range(10)]
    
    # Make predictions and check if model predicts for each image
    predictions = model.predict(np.array(test_images))
    
    # Check that predictions are made for each image
    assert predictions.shape == (10, 10)  # Assuming 10 classes for the model