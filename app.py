from flask import Flask, request, jsonify
from flasgger import Swagger  
import pandas as pd
import numpy as np
import statsmodels.api as sm


app = Flask(__name__)

# swagger config
app.config['SWAGGER'] = {
    'title': 'Population Prediction API',
    'uiversion': 3
}

swagger = Swagger(app)  


model = None

# Preprocessing/ training LM model
def preprocess_data():
    global model  
    try: # loading data via csv
        Population_df = pd.read_csv('UPDLOAD CSV DATA HERE')  # Update this path with the actual CSV path

        if Population_df.empty:
            raise ValueError("CSV data is empty.")

       
        df = Population_df[['AGE', 'SEX', 'ORIGIN', 'RACE', 'REGION', 'STATE', 'ESTIMATESBASE2020', 
                            'POPESTIMATE2020', 'POPESTIMATE2021', 'POPESTIMATE2022', 'POPESTIMATE2023']]

        # defining features for the model
        X = df.drop(columns='POPESTIMATE2023')
        y = df['POPESTIMATE2023']

        # add constant
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        return df, model  

    except Exception as e:
        print(f"Error in preprocessing and training: {str(e)}")
        raise e

# Step 6: Generate summary statistics
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
        'average_age': float(df['AGE'].mean()),
        'most_common_region': int(df['REGION'].mode()[0]),
        'most_common_state': int(df['STATE'].mode()[0]),
        'top_races': df['RACE'].value_counts().head().to_dict(),
        'top_origins': df['ORIGIN'].value_counts().head().to_dict(),
        'top_sexes': df['SEX'].value_counts().head().to_dict(),
    }
    return summary

# initialize app
@app.route('/')
def home():
    return "Welcome to the Population Prediction API!"

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
            average_age:
              type: number
            most_common_region:
              type: integer
            most_common_state:
              type: integer
            top_races:
              type: object
            top_origins:
              type: object
            top_sexes:
              type: object
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

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the population estimate for 2023 based on input parameters
    ---
    parameters:
      - name: age
        in: body
        required: true
        type: integer
      - name: sex
        in: body
        required: true
        type: string
      - name: origin
        in: body
        required: true
        type: string
      - name: race
        in: body
        required: true
        type: string
      - name: region
        in: body
        required: true
        type: string
      - name: state
        in: body
        required: true
        type: string
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

        if None in [age, sex, origin, race, region, state]:
            return jsonify({"error": "Missing or invalid required parameters"}), 400

        # Check for NaN values in the converted inputs
        if pd.isna(age) or pd.isna(sex) or pd.isna(origin) or pd.isna(race) or pd.isna(region) or pd.isna(state):
            return jsonify({"error": "Invalid numeric values for age, sex, origin, race, region, or state"}), 400

        # Create the input data
        input_data = np.array([age, sex, origin, race, region, state]).reshape(1, -1)

        # Predict the population estimate for 2023
        predicted_population = model.predict(input_data)[0]

        # Convert the predicted population to a standard int
        predicted_population = int(predicted_population)

        return jsonify({"predicted_population_estimate_2023": predicted_population})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/reload', methods=['GET'])
def reload_data():
    try:
        df, model = preprocess_data()  
        return jsonify({"message": "Data reloaded and model retrained successfully"})
    except Exception as e:
        return jsonify({"error": f"Error during data reload: {str(e)}"}), 500

@app.route('/view_data', methods=['GET'])
def view_data():
    try:
        # Load data from CSV
        Population_df = pd.read_csv('UPDLOAD CSV DATA HERE') 
        data = Population_df.to_dict(orient='records')  
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error fetching data: {str(e)}"}), 500

if __name__ == '__main__':
    with app.app_context():
        df, model = preprocess_data()  
    app.run(debug=True)
