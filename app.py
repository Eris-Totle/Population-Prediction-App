from flask import Flask, request, jsonify
from flasgger import Swagger  
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'Population Prediction API',
    'uiversion': 3
}

swagger = Swagger(app)  

model = None  

# Preprocessing/ training LM model
def preprocess_data():
    global model  
    try:
        Population_df = pd.read_csv('CSV path')  

        if Population_df.empty:
            raise ValueError("CSV data is empty.")

        df = Population_df[['AGE', 'SEX', 'ORIGIN', 'RACE', 'REGION', 'STATE', 'POPESTIMATE2023']]

        # Defining features for the model
        X = df.drop(columns='POPESTIMATE2023')  # Independent variables
        y = df['POPESTIMATE2023']  # Dependent variable

        # Training model
        model = LinearRegression()
        model.fit(X, y)  # Fit the model

        return df, model  

    except Exception as e:
        print(f"Error in preprocessing and training: {str(e)}")
        raise e

# Generate summary statistics
def generate_summary(df):
    summary = {
        'total_entries': int(len(df)),
        'average_population_estimate_2020': float(df['POPESTIMATE2020'].mean()),
        'average_population_estimate_2021': float(df['POPESTIMATE2021'].mean()),
        'average_population_estimate_2022': float(df['POPESTIMATE2022'].mean()),
        'average_population_estimate_2023': float(df['POPESTIMATE2023'].mean()),
        'min_population_estimate_2020': int(df['POPESTIMATE2020'].min()),
        'max_population_estimate_2020': int(df['POPESTIMATE2020'].max()),
        'min_population_estimate_2021': int(df['POPESTIMATE2021'].min()),
        'max_population_estimate_2021': int(df['POPESTIMATE2021'].max()),
        'min_population_estimate_2022': int(df['POPESTIMATE2022'].min()),
        'max_population_estimate_2022': int(df['POPESTIMATE2022'].max()),
        'min_population_estimate_2023': int(df['POPESTIMATE2023'].min()),
        'max_population_estimate_2023': int(df['POPESTIMATE2023'].max()),
    }
    return summary

# Home route
@app.route('/')
def home():
    return "Welcome to the Population Prediction API!"

# Summary statistics route
@app.route('/summary', methods=['GET'])
def get_summary():
    '''
    Generate summary statistics for the population dataset
    ---
    responses:
      200:
        description: Population summary statistics
        schema:
          type: object
          properties:
            total_entries:
              type: integer
            average_population_estimate_2020:
              type: number
            average_population_estimate_2021:
              type: number
            average_population_estimate_2022:
              type: number
            average_population_estimate_2023:
              type: number
            min_population_estimate_2020:
              type: integer
            max_population_estimate_2020:
              type: integer
            min_population_estimate_2021:
              type: integer
            max_population_estimate_2021:
              type: integer
            min_population_estimate_2022:
              type: integer
            max_population_estimate_2022:
              type: integer
            min_population_estimate_2023:
              type: integer
            max_population_estimate_2023:
              type: integer
    '''
    try:
        # Load data and generate summary statistics
        df, _ = preprocess_data()
        summary = generate_summary(df)
        return jsonify(summary)
    except ValueError as e:
        return jsonify({"error": f"ValueError: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the population estimate for 2023 based on input parameters.
    ---
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            age:
              type: integer
              description: Age of the individual (between 0 and 85)
            sex:
              type: integer
              enum: [0, 1, 2]  
              description: Sex of the individual
            origin:
              type: integer
              enum: [0, 1, 2]  
              description: Origin of the individual
            race:
              type: integer
              enum: [1, 2, 3, 4, 5, 6]  # Different race categories
              description: Race of the individual
            region:
              type: integer
              enum: [1, 2, 3, 4]  # Example regions
              description: Region of the individual
            state:
              type: integer
              description: State (from 0 to 56)
              example: 10  # For example, a state code
    responses:
      200:
        description: Predicted population estimate for 2023
        schema:
          type: object
          properties:
            predicted_population_estimate_2023:
              type: integer
              description: Predicted population estimate
    '''
    global model
    if model is None:
        return jsonify({"error": "The model has not been trained. Please refresh the data by calling the '/reload' endpoint first."}), 400

    data = request.json
    try:
        # Extract input values
        age = data.get('age')
        sex = data.get('sex')
        origin = data.get('origin')
        race = data.get('race')
        region = data.get('region')
        state = data.get('state')

        # Check for missing or invalid parameters
        if None in [age, sex, origin, race, region, state]:
            return jsonify({"error": "Missing or invalid required parameters"}), 400
        
        # Convert to appropriate types (ensure everything is numeric)
        try:
            age = int(age)
            sex = int(sex)
            origin = int(origin)
            race = int(race)
            region = int(region)
            state = int(state)
        except ValueError:
            return jsonify({"error": "All input values must be valid integers"}), 400

        # Ensuring no Nan
        if pd.isna(age) or pd.isna(sex) or pd.isna(origin) or pd.isna(race) or pd.isna(region) or pd.isna(state):
            return jsonify({"error": "Invalid numeric values for age, sex, origin, race, region, or state"}), 400

        # Create the input data as a numpy array
        input_data = np.array([age, sex, origin, race, region, state]).reshape(1, -1)

        # Predict the population estimate for 2023 using the model
        predicted_population = model.predict(input_data)[0]

        # Converting prediction output to int
        predicted_population = int(predicted_population)

        return jsonify({"predicted_population_estimate_2023": predicted_population})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Reload data and retrain model route
@app.route('/reload', methods=['GET'])
def reload_data():
    try:
        global model  
        df, model = preprocess_data() 
        return jsonify({"message": "Data reloaded and model retrained successfully"})
    except Exception as e:
        return jsonify({"error": f"Error during data reload: {str(e)}"}), 500

# View data route
@app.route('/view_data', methods=['GET'])
def view_data():
    try:
        Population_df = pd.read_csv('CSV path') 
        data = Population_df.to_dict(orient='records')  
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error fetching data: {str(e)}"}), 500

if __name__ == '__main__':
    with app.app_context():
        df, model = preprocess_data()  # Preprocess and load the model when starting the app
    app.run(debug=True)
