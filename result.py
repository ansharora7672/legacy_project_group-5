import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.models import load_model


# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

# Class names for CIFAR-10
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display some sample images from the dataset
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()

# Constructing neural net


# Load the pre-trained model

model = load_model('model/image_classifier.keras')

# Load and preprocess the image to make a prediction

img = cv.imread("car.png")

img = cv.resize(img, (32, 32))  # Resize the image to match CIFAR-10 image size
img = img / 255.00  # Normalize the pixel values to be between 0 and 1
img=img[None,:]

# Make a prediction
prediction = model.predict(img)
index = np.argmax(prediction)
print(f"Prediction is: {class_names[index]}")
