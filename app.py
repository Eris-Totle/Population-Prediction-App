from flask import Flask, request, jsonify
from flasgger import Swagger  
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
        Population_df = pd.read_csv('/Users/erisdodds/Documents/sc-est2023-alldata6 (1).csv', dtype={'STATE': str})  

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

# Load dataset globally
df, model = preprocess_data()

# home route
@app.route('/')
def home():
    """
    Welcome to the Population Prediction API!
    ---
    responses:
      200:
        description: API is running successfully
    """
    return "Welcome to the Population Prediction API!"

# 📌 **Summary Statistics Route**
@app.route('/summary', methods=['GET'])
def get_summary():
    """
    Generate summary statistics for the population dataset
    ---
    responses:
      200:
        description: Population summary statistics
    """
    try:
        summary = {
            'total_entries': int(len(df)),
            'average_population_estimate_2023': float(df['POPESTIMATE2023'].mean()),
            'min_population_estimate_2023': int(df['POPESTIMATE2023'].min()),
            'max_population_estimate_2023': int(df['POPESTIMATE2023'].max()),
        }
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/choropleth', methods=['GET'])
def get_choropleth():
    """
    Generate a choropleth map based on user-selected parameters (age, sex, origin, race).
    ---
    parameters:
      - in: query
        name: age
        type: integer
        description: Age filter (optional)
      - in: query
        name: sex
        type: integer
        enum: [0, 1, 2]  
        description: Sex filter (optional)
      - in: query
        name: origin
        type: integer
        enum: [0, 1, 2]
        description: Origin filter (optional)
      - in: query
        name: race
        type: integer
        enum: [1, 2, 3, 4, 5, 6]
        description: Race filter (optional)
    responses:
      200:
        description: Returns a choropleth map as JSON
    """
    try:
        filtered_df = df.copy()

        # Apply filters
        age = request.args.get('age', type=int)
        sex = request.args.get('sex', type=int)
        origin = request.args.get('origin', type=int)
        race = request.args.get('race', type=int)

        if age is not None:
            filtered_df = filtered_df[filtered_df['AGE'] == age]
        if sex is not None:
            filtered_df = filtered_df[filtered_df['SEX'] == sex]
        if origin is not None:
            filtered_df = filtered_df[filtered_df['ORIGIN'] == origin]
        if race is not None:
            filtered_df = filtered_df[filtered_df['RACE'] == race]

     
        state_population = filtered_df.groupby('STATE', as_index=False)['POPESTIMATE2023'].sum()

        state_population['POPESTIMATE2023'] = state_population['POPESTIMATE2023'].astype(int)

        # Generate choropleth map
        fig = go.Figure(data=go.Choropleth(
            locations=state_population['STATE'],  
            z=state_population['POPESTIMATE2023'],  
            locationmode='USA-states',  
            colorscale='Reds',  
            colorbar_title="Population",  
        ))

        fig.update_layout(
            title_text='Filtered US Population Estimate (2023)',  
            geo_scope='usa',  
        )

        # Return JSON with proper int conversion
        graph_json = fig.to_json()
        return jsonify({
            "choropleth_map": graph_json,
            "filtered_population_sum": int(state_population['POPESTIMATE2023'].sum())  # Convert int64 to int
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the population estimate for 2023 based on input parameters.
    """
    global model
    if model is None:
        return jsonify({"error": "Model not trained. Use '/reload'."}), 400

    data = request.json
    try:
        input_data = np.array([data.get('age'), data.get('sex'), data.get('origin'),
                               data.get('race'), data.get('region'), data.get('state')]).reshape(1, -1)

        predicted_population = int(model.predict(input_data)[0])

        return jsonify({"predicted_population_estimate_2023": predicted_population})
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/reload', methods=['GET'])
def reload_data():
    """
    Reload the dataset and retrain the model.
    ---
    responses:
      200:
        description: Data reloaded successfully
    """
    try:
        global df, model  
        df, model = preprocess_data() 
        return jsonify({"message": "Data reloaded and model retrained successfully"})
    except Exception as e:
        return jsonify({"error": f"Error during data reload: {str(e)}"}), 500

@app.route('/view_data', methods=['GET'])
def view_data():
    """
    View the raw dataset.
    ---
    responses:
      200:
        description: Returns raw dataset
    """
    try:
        data = df.to_dict(orient='records')  
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error fetching data: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
