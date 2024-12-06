import os
import sys
import shutil
import pytest
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app import app
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
@pytest.fixture

def client():
    """Create a Flask test client"""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    return app.test_client()

@pytest.fixture
def test_image():
    """Create a test image for uploads"""
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Create a test image (32x32 red square)
    test_image_path = 'test_image.png'
    img = Image.new('RGB', (32, 32), color='red')
    img.save(test_image_path)
    
    yield test_image_path
    
    # Cleanup after test
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    if os.path.exists('uploads'):
        shutil.rmtree('uploads')

def test_homepage_loads(client):
    """Test that the homepage loads correctly"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data
    assert b'form' in response.data.lower()

def test_image_upload_and_classification(client, test_image):
    """Test image upload and classification workflow"""
    # Open test image
    with open(test_image, 'rb') as img:
        # Create test data
        data = {
            'file': (img, 'test_image.png')
        }
        
        # Submit POST request with test image
        response = client.post('/', 
                               content_type='multipart/form-data',
                               data=data,
                               follow_redirects=True)
        
        # Check response
        assert response.status_code == 200
        
        # Verify prediction
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        response_text = response.data.decode('utf-8').lower()
        prediction_found = any(class_name in response_text 
                               for class_name in class_names)
        assert prediction_found, "No prediction found in response"
        
        # Verify uploaded image saved
        assert os.path.exists('uploads/uploaded_image.png'), \
            "Uploaded image was not saved"