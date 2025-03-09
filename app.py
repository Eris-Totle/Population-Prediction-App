import folium
from flask import render_template_string
from folium.plugins import HeatMap
from flask import Flask, request, jsonify
from flasgger import Swagger  
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import branca
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Swagger config
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

        state_mapping = {
            '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE', '12': 'FL',
            '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY',
            '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO',
            '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC',
            '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD',
            '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
            '56': 'WY'
        }
        Population_df['STATE_ABBR'] = Population_df['STATE'].map(state_mapping)
        df = Population_df[['AGE', 'SEX', 'ORIGIN', 'RACE', 'REGION', 'STATE', 'STATE_ABBR', 'POPESTIMATE2023']]

        
        X = df.drop(columns=['POPESTIMATE2023', 'STATE_ABBR'])  
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

@app.route('/heatmap', methods=['GET'])
def get_heatmap():
    """ Generate a Folium Heatmap based on user-selected parameters (age, sex, origin, race). """

    try:
        age = request.args.get('age', type=int)
        sex = request.args.get('sex', type=int)
        origin = request.args.get('origin', type=int)
        race = request.args.get('race', type=int)

        filtered_df = df.copy()
        if age is not None:
            filtered_df = filtered_df[filtered_df['AGE'] == age]
        if sex is not None:
            filtered_df = filtered_df[filtered_df['SEX'] == sex]
        if origin is not None:
            filtered_df = filtered_df[filtered_df['ORIGIN'] == origin]
        if race is not None:
            filtered_df = filtered_df[filtered_df['RACE'] == race]

        
        state_population = filtered_df.groupby('STATE_ABBR', as_index=False)['POPESTIMATE2023'].sum()

        if state_population.empty:
            return jsonify({"error": "No data available for the selected filters."})

        state_coords = {
            'AL': [32.806671, -86.791130], 'AK': [61.370716, -152.404419], 'AZ': [33.729759, -111.431221],
            'AR': [34.969704, -92.373123], 'CA': [36.116203, -119.681564], 'CO': [39.059811, -105.311104],
            'CT': [41.597782, -72.755371], 'DE': [39.318523, -75.507141], 'FL': [27.766279, -81.686783],
            'GA': [33.040619, -83.643074], 'HI': [21.094318, -157.498337], 'ID': [44.240459, -114.478828],
            'IL': [40.349457, -88.986137], 'IN': [39.849426, -86.258278], 'IA': [42.011539, -93.210526],
            'KS': [38.526600, -96.726486], 'KY': [37.668140, -84.670067], 'LA': [31.169546, -91.867805],
            'ME': [44.693947, -69.381927], 'MD': [39.063946, -76.802101], 'MA': [42.230171, -71.530106],
            'MI': [43.326618, -84.536095], 'MN': [45.694454, -93.900192], 'MS': [32.741646, -89.678696],
            'MO': [38.456085, -92.288368], 'MT': [46.921925, -110.454353], 'NE': [41.125370, -98.268082],
            'NV': [38.313515, -117.055374], 'NH': [43.452492, -71.563896], 'NJ': [40.298904, -74.521011],
            'NM': [34.840515, -106.248482], 'NY': [42.165726, -74.948051], 'NC': [35.630066, -79.806419],
            'ND': [47.528912, -99.784012], 'OH': [40.388783, -82.764915], 'OK': [35.565342, -96.928917],
            'OR': [44.572021, -122.070938], 'PA': [40.590752, -77.209755], 'RI': [41.680893, -71.511780],
            'SC': [33.856892, -80.945007], 'SD': [44.299782, -99.438828], 'TN': [35.747845, -86.692345],
            'TX': [31.054487, -97.563461], 'UT': [40.150032, -111.862434], 'VT': [44.045876, -72.710686],
            'VA': [37.769337, -78.169968], 'WA': [47.400902, -121.490494], 'WV': [38.491226, -80.954456],
            'WI': [44.268543, -89.616508], 'WY': [42.755966, -107.302490]
        }

        us_map = folium.Map(location=[37.8, -96], zoom_start=5)

        heat_data = [[state_coords[state][0], state_coords[state][1], pop]
                     for state, pop in zip(state_population['STATE_ABBR'], state_population['POPESTIMATE2023'])
                     if state in state_coords]
        HeatMap(heat_data, radius=25, blur=15, max_zoom=1).add_to(us_map)

        colormap = branca.colormap.LinearColormap(colors=['blue', 'green', 'yellow', 'red'],
                                                  vmin=state_population['POPESTIMATE2023'].min(),
                                                  vmax=state_population['POPESTIMATE2023'].max())
        colormap.caption = "Population Intensity (2023)"
        colormap.add_to(us_map)

        for _, row in state_population.iterrows():
            state = row['STATE_ABBR']
            pop = row['POPESTIMATE2023']
            if state in state_coords:
              folium.CircleMarker(
                  location=state_coords[state],
                  radius=10,  
                  color="transparent",  
                  fill=True,
                  fill_color="transparent",  
                  fill_opacity=0,  
                  opacity=0,  
                  tooltip=f"<b>{state}</b>: {pop:,} people"  
              ).add_to(us_map)

        us_map.get_root().render()
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>{{ us_map.get_root().header.render()|safe }}</head>
        <body>
            <h1>Population Heatmap (2023)</h1>
            <form method="GET" action="/heatmap">
                Age: <input type="number" name="age">
                Sex: <input type="number" name="sex">
                Origin: <input type="number" name="origin">
                Race: <input type="number" name="race">
                <button type="submit">Filter</button>
            </form>
            {{ us_map.get_root().html.render()|safe }}
            <script>{{ us_map.get_root().script.render()|safe }}</script>
        </body>
        </html>""", us_map=us_map)

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
from flasgger import swag_from

from flasgger import swag_from

@app.route('/predict', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'age': {'type': 'integer', 'description': 'Age filter (optional)'},
                    'sex': {'type': 'integer', 'enum': [0, 1, 2], 'description': 'Sex filter (optional)'},
                    'origin': {'type': 'integer', 'enum': [0, 1, 2], 'description': 'Origin filter (optional)'},
                    'race': {'type': 'integer', 'enum': [1, 2, 3, 4, 5, 6], 'description': 'Race filter (optional)'},
                    'region': {'type': 'integer', 'description': 'Region filter (optional)'},
                    'state': {'type': 'integer', 'description': 'State filter (optional)'}
                }
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Predicted population estimate for 2023',
            'schema': {
                'type': 'object',
                'properties': {
                    'predicted_population_estimate_2023': {
                        'type': 'integer',
                        'description': 'Predicted population estimate'
                    }
                }
            }
        },
        400: {
            'description': 'Bad Request - Invalid input or missing model'
        },
        415: {
            'description': 'Unsupported Media Type - Ensure Content-Type is application/json'
        }
    }
})
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model not trained. Use '/reload'."}), 400


    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type. Make sure to send JSON data with 'Content-Type: application/json'"}), 415

    data = request.get_json()

    try:
      
        age = data.get('age')
        sex = data.get('sex')
        origin = data.get('origin')
        race = data.get('race')
        region = data.get('region')
        state = data.get('state')

       
        if None in [age, sex, origin, race, region, state]:
            return jsonify({"error": "Missing required input fields"}), 400

      
        input_data = np.array([age, sex, origin, race, region, state]).reshape(1, -1)

   
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
