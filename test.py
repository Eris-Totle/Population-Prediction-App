import pytest
from app import app

# Define valid input for prediction test
valid_input = {
    "age": 2,
    "sex": 1,
    "origin": 1,
    "race": 1,
    "region": 2,
    "state": 15
}

# Define missing input for prediction test
missing_input = {
    "age": 2,
    "sex": 1,
    "origin": 1,
    "state": 15
}

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context(): 
            yield client

def test_reload_data(client):
    """Test the reload endpoint that loads the data."""
    response = client.get('/reload')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'message' in json_data
    assert json_data['message'] == 'Data reloaded and model retrained successfully'

def test_predict_after_reload(client):
    """Test prediction endpoint after reloading the data."""
    client.get('/reload')  

    # Test valid prediction
    response = client.post('/predict', json=valid_input)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'predicted_population_estimate_2023' in json_data

def test_missing_fields(client):
    """Test prediction with missing fields."""
    # Reload the data first
    client.get('/reload')  
    
    # Test with missing fields
    response = client.post('/predict', json=missing_input)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Missing required input fields" in json_data['error']
