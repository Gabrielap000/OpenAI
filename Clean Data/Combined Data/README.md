Two-Layer Prediction Model

Layer 1: Weather Prediction Model
  Input: Latitude and Longitude
  Output: Predicted weather variables (e.g., air temperature, relative humidity, wind speed, wind direction, solar flux)

Layer 2: UHI Prediction Model
  Input: Predicted weather variables from Layer 1
  Output: Predicted UHI index

Steps of code:

Load the datasets: Load the UHI index data and weather data.
Calculate distances: Calculate the distance from each data point to the Bronx and Manhattan stations.
Interpolate weather data: Interpolate the weather data for each data point based on the distances to the two stations.
Train-Test Split: Split the data into training and testing sets.
Train Weather Prediction Model: Train a model to predict weather data based on latitude and longitude.
Train UHI Prediction Model: Train a model to predict the UHI index based on the predicted weather data.
Evaluate the Models: Evaluate the performance of both models.
