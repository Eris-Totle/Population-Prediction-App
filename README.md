# Population Prediction API
 This is an experiment with a Flask-based API, which predicts populations given specific parameters of interest (i.e. state, region, sex, race).
 The API has four main endpoints
 * /predict Predicts the population given specified parameters.
 * /summary Summarizes statistical dimensions of the dataset.

## Data Source and Prediction Process
The data source comes from within the [U.S. Census Bureau - Population Estimates Data](https://www.census.gov/programs-surveys/popest/data/data-sets.html). This data contains population estimates for 6 race groups by sex, age, and hispanic origin. Use the data summary [DATA_key](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2023/SC-EST2023-ALLDATA6.pdf?utm_source=chatgpt.com) to understand the paramaters/ keys corresponding to numerical values in columns. The data includes features including:
* SUMLEV
* REGION
* DIVISION
* STATE
* NAME
* SEX
* ORIGIN
* RACE
* AGE

The complete dataset can be downloaded [Here](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2023/)

### Prediction Process
The application uses a simple linear regression model to make population predictions. The input features are limited to 6 input features including race, sex, state, region, age, origin. The prediction process follows 3 steps:

1) **Data Preprocessing:** Data loaded, columns selected/ removed.
2) **Model Trained** A linear regression model is trained to predictor variables including demographics and regions. Target variable is the population estimate for 2023.
3) **Prediction** The model then makes predictions based on the input features/ parameters input by the user.

Using this model allows users to get specific population estimates based on demographic and regional features of interest. 

## Pre-Requisites 
...
