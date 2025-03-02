from flask import Flask, request, jsonify, render_template
from flasgger import Swagger  
import pandas as pd
import numpy as np
import plotly  # ✅ Import plotly
import plotly.graph_objects as go
import plotly.utils
import json  
from sklearn.linear_model import LinearRegression

app = Flask(__name__)


app.config['SWAGGER'] = {
    'title': 'Population Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)


model = None  

def preprocess_data():
    global model  
    try:
        Population_df = pd.read_csv('/Users/erisdodds/Documents/sc-est2023-alldata6 (1).csv', dtype={'STATE': str})  

        if Population_df.empty:
            raise ValueError("CSV data is empty.")

        df = Population_df[['AGE', 'SEX', 'ORIGIN', 'RACE', 'REGION', 'STATE', 'POPESTIMATE2023']]

        # Define Features & Train Model
        X = df.drop(columns='POPESTIMATE2023')
        y = df['POPESTIMATE2023']

        model = LinearRegression()
        model.fit(X, y)  

        return df, model  
    except Exception as e:
        print(f"Error in preprocessing and training: {str(e)}")
        raise e


df, model = preprocess_data()


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
    try:
        filtered_df = df.copy()

      
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

        if state_population.empty:
            print("\n--- DEBUG: No Data Available for Selected Filters ---\n")
            return jsonify({"error": "No data available for the selected filters."})

        state_population['POPESTIMATE2023'] = state_population['POPESTIMATE2023'].astype(int)

        fig = go.Figure(data=go.Choropleth(
            locations=state_population['STATE'].astype(str),  
            z=state_population['POPESTIMATE2023'],  
            locationmode='USA-states',  
            colorscale='Reds',  
            colorbar_title="Population",  
        ))

        fig.update_layout(
            title_text='Filtered US Population Estimate (2023)',  
            geo_scope='usa',
            height=600
        )


        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        print("\n--- DEBUG: Graph JSON ---")
        print(graph_json[:500])  
        print("--- END DEBUG ---\n")

        return render_template("choropleth.html", graph_json=graph_json, age=age, sex=sex, origin=origin, race=race)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/choropleth', methods=['GET'])
def get_choropleth():
    try:
        filtered_df = df.copy()


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

        if state_population.empty:
            print("\n--- DEBUG: No Data Available for Selected Filters ---\n")
            return jsonify({"error": "No data available for the selected filters."})


        state_population['POPESTIMATE2023'] = state_population['POPESTIMATE2023'].astype(int)


        fig = go.Figure(data=go.Choropleth(
            locations=state_population['STATE'].astype(str),  
            z=state_population['POPESTIMATE2023'],  
            locationmode='USA-states',  
            colorscale='Reds',  
            colorbar_title="Population",  
        ))

        fig.update_layout(
            title_text='Filtered US Population Estimate (2023)',  
            geo_scope='usa',
            height=600
        )

  
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        print("\n--- DEBUG: Graph JSON ---")
        print(graph_json[:500])  
        print("--- END DEBUG ---\n")

        return render_template("choropleth.html", graph_json=graph_json, age=age, sex=sex, origin=origin, race=race)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(debug=True)

