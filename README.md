# Image-Classifier
# Image Classification with TensorFlow and Flask

This repository contains code for a simple image classification project using a Convolutional Neural Network (CNN) created with TensorFlow. The project involves training the model on the CIFAR-10 dataset and deploying it in a Flask web application.

## Prerequisites

Make sure you have the following libraries installed:

- numpy
- OpenCV (cv2)
- matplotlib
- TensorFlow
- Flask

Install the required libraries using:

```bash
pip install numpy opencv-python matplotlib tensorflow flask
```

## Usage

1. **Training the Model:**
   - Run `model_creation.py` to create and train the CNN model on the CIFAR-10 dataset.

   ```bash
   python model_creation.py
   ```

2. **Web Application:**
   - The trained model is used in `app.py` for creating a Flask web application.
   - Run the Flask app:

   ```bash
   python app.py
   ```

   Visit `http://localhost:5000` in your browser and upload an image to get a classification result.

3. **Result Analysis:**
   - After running the web application, the uploaded image's classification result is analyzed in `result.py`.

   ```bash
   python result.py
   ```

   This script loads the trained model and preprocesses an example image, showcasing the classification result.

## Directory Structure

- **`uploads/`:** Contains uploaded images from the web application.
- **`model/`:** Stores the trained model (`image_classifier.model`).
- **`templates/`:** Contains HTML templates for the Flask web application.
- **`app.py`:** Flask web application script.
- **`model_creation.py`:** Script for creating and training the CNN model.
- **`result.py`:** Script for analyzing the model's classification result.

## Contributing

 Your feedback and suggestions are highly appreciated!

## License

This project is licensed under the [MIT License](LICENSE).
