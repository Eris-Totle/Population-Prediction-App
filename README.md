# Population Prediction API
 This is an experiment with a Flask-based API, which predicts populations given specific parameters of interest (i.e. state, region, sex, race).
 The API has four main endpoints
 * /predict Predicts the population given specified parameters.
 * /summary Summarizes statistical dimensions of the dataset.
 * /heatmap A heatmap displaying population across states by selectable parameters (i.e. race, sex, gender)

## Data Source and Prediction Process
The data source comes from within the [U.S. Census Bureau - Population Estimates Data](https://www.census.gov/programs-surveys/popest/data/data-sets.html). This data contains population estimates for 6 race groups by sex, age, and hispanic origin. Use the data summary [DATA_key](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2023/SC-EST2023-ALLDATA6.pdf?utm_source=chatgpt.com) to understand the paramaters/ keys corresponding to numerical values in columns. The data includes features including: SUMLEV, REGION, DIVISION, STATE, NAME, SEX, ORIGIN, RACE, AGE.

The complete dataset can be downloaded [Here](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2023/)

### Prediction Process
The application allows the user to choose between prediction models (i.e.....) . The input features are limited to 6 input features including race, sex, state, region, age, origin. The prediction process follows 3 steps:

1) **Data Preprocessing:** Data loaded, columns selected/ removed.
2) **Model Trained** A linear regression model is trained to predictor variables including demographics and regions. Target variable is the population estimate for 2023.
3) **Prediction** The model then makes predictions based on the input features/ parameters input by the user.

Using this model allows users to get specific population estimates based on demographic and regional features of interest. 

### Heatmap
The application includes a folium heapmap with inline html rendering to produce a heatmap of populations per selected parameters of a users interest. This resource was helpful for developing this map type with inline html rendering [Mapping Guide](https://python-visualization.github.io/folium/latest/advanced_guide/flask.html). While the same could probably be achieved through sending the data to Power BI, it was neat to see how parameters/ features of the resulting map could be customized using flask/ folium libraries.

#### Sample Visual
<img width="1195" alt="Screenshot 2025-03-13 at 5 23 47 PM" src="https://github.com/user-attachments/assets/de62082b-e52a-4cde-99a9-91771ea43853" />

## Pre-Requisites 

Running the app requires the following:

Python 3.9+
pip (Python package installer)

Further, 
Virtualenv (Optional but recommended)

## Setting Up on Mac/OS

### 1. Create Text File

Create a text file named app.py, ideally in a place easy to navigate to on your local machine. Use the code provided in the app.py provided here, modifying the script to include the CSV file for the data provided. 

Once the file is saved, open your terminal to navigate to the directory of the project (i.e. cd /Users/Username/Documents/App_project)

### 2. Create a Virtual Environment (Optional)

Create a virtual environment prior to installing depenencies needed for this project: 

For MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```
For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install the Dependencies
Intall dependencies for this project using pip

```bash
pip install -r requirements.txt
```
### 4. Environmental Variables
Prior to running the app, you'll need to creat a .env file in the project root. Add the following:

```bash
FLASK_APP=app.py
FLASK_ENV=development
```
Setting up environmental variables in OS:
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
```

In Windows powershell

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
```
### 5. Running the application

When you're ready to run the app
```bash
flask run
```
### 6. Swagger Documentation

The Swagger documentation and some feature of the app will be available at

```bash
http://127.0.0.1:5000/apidocs/
```
### 7. Testing Endpoints

Reloading the data/ training the model can be done through the /reload endpoint
```bash
curl -X POST http://127.0.0.1:5000/reload
```

Predicting data can be done through inputting parameters (use the data key for guidance here)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 16,
  "origin": 1,
  "race": 1,
  "region": 1,
  "sex": 1,
  "state": 17
}'

### 8. Stopping the Application
To shop the application - Ctrl + C

## Running Tests - In progress
Pytests 

## Deploying to Heroku


