# GetMeThere
Insight Project: Help people decide whether they should take a cab or a metro to travel in NYC

Description of the app:
Enter an origin and destination and get the cab fare per person, the cab travel time, metro ticket cost and metro transit time (including the walking time to and from the metro station).
Information about the weather is given
Additionally, you can input a depart time and a number of people. If not, it is assumed that a single person is inquiring for an immediate time.

The files and folders necessary to run the app are available.

To run the app locally, please run run.py
The fits are in the folder Regression and are loaded with joblib, with scikit-learn 0.16.1

app folder
  - PredictFare.py: file making the cab fare and time travel prediction
  - view.py: architecture of the app
  - featureNames.csv: needed for the pandas dataframe in PredictFare.py
  - Regression: folder containing the fits
  - LoadFiles: various files to be loaded for the fit, in PredictFare.py
  - static: css, js, fonts and Images used in
