#code taken from - https://flask.palletsprojects.com/en/stable/testing/
# https://gist.github.com/justinmklam/c453ea62b47f4e302dbd83c05882e3bd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pytest
from app import app
from io import BytesIO
import cv2 as cv
import numpy as np


@pytest.fixture()
def client():
        app.config.update({
        "TESTING": True,
    })
        with app.test_client() as client:

            yield client



def test_request(client):
    response = client.put("/")
    assert b"<title>Image Classifier</title>" in response.data

def test_post_without_file(client):
    
    response = client.post("/", data={})
    assert response.status_code == 400
    assert b"Bad Request" in response.data

@pytest.fixture(scope='module')
def test_client():
    """Provide a test client for the Flask app."""
    with app.test_client() as client:
        yield client

@pytest.fixture(scope='function')
def test_upload_dir():
    """Set up and clean up the upload directory."""
    test_dir = 'uploads'
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    try:
        os.rmdir(test_dir)
    except OSError:
        pass  # Directory not empty or already deleted

def generate_test_image(size, color_space, noise_factor=0.0):
    """Generate a test image with specific characteristics."""
    if color_space == 'RGB':
        img = np.random.rand(size[0], size[1], 3) * 255
    else:  # Grayscale
        img = np.random.rand(size[0], size[1]) * 255
        img = np.stack((img,) * 3, axis=-1)  # Convert to 3 channels

    if noise_factor > 0:
        noise = np.random.normal(0, noise_factor, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255)

    return img.astype(np.uint8)

def test_concurrent_mixed_format_uploads(test_client, test_upload_dir):
    """Test concurrent uploads with varying image characteristics."""
    test_configs = [
        {'size': (32, 32), 'color_space': 'RGB', 'noise': 0.0, 'name': 'standard'},
        {'size': (32, 32), 'color_space': 'RGB', 'noise': 0.2, 'name': 'noisy'},
        {'size': (32, 32), 'color_space': 'Grayscale', 'noise': 0.0, 'name': 'gray'},
        {'size': (32, 32), 'color_space': 'RGB', 'noise': 0.1, 'name': 'noise_medium'}
    ]

    results = []

    for config in test_configs:
        # Generate test image
        test_img = generate_test_image(
            config['size'],
            config['color_space'],
            config['noise']
        )

        # Save test image
        temp_path = os.path.join(test_upload_dir, f"test_{config['name']}.png")
        cv.imwrite(temp_path, test_img)

        # Verify the saved image
        saved_test_img = cv.imread(temp_path)
        assert saved_test_img is not None, f"Failed to save test image for {config['name']}"

        # Test image upload and classification
        with open(temp_path, 'rb') as img_file:
            image_binary = img_file.read()

            response = test_client.post(
                '/',
                data={'file': (BytesIO(image_binary), f"test_{config['name']}.png")},
                content_type='multipart/form-data'
            )

        # Verify response
        assert response.status_code == 200
        assert b'<h2>Result:' in response.data
        assert b'<img src="uploads/uploaded_image.png"' in response.data

        # Verify processed image
        assert os.path.exists('uploads/uploaded_image.png')
        processed_img = cv.imread('uploads/uploaded_image.png')
        assert processed_img is not None, "Failed to load processed image"
        height, width = processed_img.shape[:2]
        assert height == 32, f"Image height should be 32, got {height}"
        assert width == 32, f"Image width should be 32, got {width}"

        results.append({
            'config': config,
            'prediction_success': b'Result:' in response.data,
            'template_rendered': b'<!DOCTYPE html>' in response.data,
            'image_processed': processed_img is not None,
            'image_size': processed_img.shape[:2]
        })

        # Clean up processed image
        os.remove('uploads/uploaded_image.png')

    # Analyze results
    assert len(results) == len(test_configs)
    for result in results:
        assert result['prediction_success']
        assert result['template_rendered']
        assert result['image_processed']
        assert result['image_size'] == (32, 32)

    # Clean up test images
    for config in test_configs:
        temp_path = os.path.join(test_upload_dir, f"test_{config['name']}.png")
        if os.path.exists(temp_path):
            os.remove(temp_path)