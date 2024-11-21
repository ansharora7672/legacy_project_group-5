#code taken from - https://flask.palletsprojects.com/en/stable/testing/
# https://gist.github.com/justinmklam/c453ea62b47f4e302dbd83c05882e3bd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pytest
from app import app


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