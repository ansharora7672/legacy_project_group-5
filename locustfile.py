from locust import HttpUser, task, between
import io
from PIL import Image
import numpy as np

# Define a Locust user class-----

class UserBehavior(HttpUser):
    wait_time = between(1, 3)  # Time between user actions (1 to 3 seconds)

    @task
    def load_home_page(self):
        response = self.client.get("/")
        assert response.status_code == 200

    @task
    def upload_image(self):
        # Create a mock image for testing
        img = Image.new("RGB", (32, 32), color="blue")  # Create a 32x32 blue image
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')  # Save image to a byte stream
        img_io.seek(0)  # Reset the stream position to the beginning

        # Prepare the form data with the image file
        data = {
            'file': (img_io, 'test_image.png')
        }

        # Send a POST request to the '/' endpoint to upload the image
        response = self.client.post("/", files=data, headers={"Content-Type": "multipart/form-data"})

        # Assert that the prediction is returned in the response
        assert response.status_code == 200
        assert b"prediction" in response.data  # Look for prediction in the response

    @task
    def upload_multiple_images(self):
        for i in range(3):  # For example, upload 3 images
            # Create a mock image for testing
            img = Image.new("RGB", (32, 32), color="red")  # Create a 32x32 red image
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')  # Save image to a byte stream
            img_io.seek(0)  # Reset the stream position to the beginning

            # Prepare the form data with the image file
            data = {
                'file': (img_io, f'test_image_{i}.png')  # Name the file uniquely
            }

            # Send a POST request to the '/' endpoint to upload the image
            response = self.client.post("/", files=data, headers={"Content-Type": "multipart/form-data"})

            # Assert that the prediction is returned in the response
            assert response.status_code == 200
            assert b"prediction" in response.data  # Look for prediction in the response
