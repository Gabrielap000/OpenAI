
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the UHI index data
uhi_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')

# Load the weather data
weather_bronx = pd.read_excel('weather_data.xlsx', sheet_name='Bronx')
weather_manhattan = pd.read_excel('weather_data.xlsx', sheet_name='Manhattan')

# Convert time columns to datetime
uhi_data['datetime'] = pd.to_datetime(uhi_data['datetime'])
weather_bronx['datetime'] = pd.to_datetime(weather_bronx['Date / Time'])
weather_manhattan['datetime'] = pd.to_datetime(weather_manhattan['Date / Time'])

# Define the coordinates of the Bronx and Manhattan stations
bronx_coords = (40.87248, -73.89352)
manhattan_coords = (40.76754, -73.96449)

# Calculate the distance from each data point to the Bronx and Manhattan stations
uhi_data['distance_bronx'] = uhi_data.apply(lambda row: geodesic((row['Latitude'], row['Longitude']), bronx_coords).meters, axis=1)
uhi_data['distance_manhattan'] = uhi_data.apply(lambda row: geodesic((row['Latitude'], row['Longitude']), manhattan_coords).meters, axis=1)

# Function to interpolate weather data based on distances
def interpolate_weather(row, weather_bronx, weather_manhattan):
    time = row['datetime']
    bronx_weather = weather_bronx.iloc[(weather_bronx['datetime'] - time).abs().argsort()[:1]]
    manhattan_weather = weather_manhattan.iloc[(weather_manhattan['datetime'] - time).abs().argsort()[:1]]
    
    distance_bronx = row['distance_bronx']
    distance_manhattan = row['distance_manhattan']
    total_distance = distance_bronx + distance_manhattan
    
    weight_bronx = distance_manhattan / total_distance
    weight_manhattan = distance_bronx / total_distance
    
    interpolated_weather = {}
    for col in ['Air Temp at Surface [degC]', 'Relative Humidity [percent]', 'Avg Wind Speed [m/s]', 'Wind Direction [degrees]', 'Solar Flux [W/m^2]']:
        interpolated_weather[col] = (weight_bronx * bronx_weather[col].values[0] + weight_manhattan * manhattan_weather[col].values[0])
    
    return pd.Series(interpolated_weather)

# Interpolate weather data for each data point
interpolated_weather_data = uhi_data.apply(interpolate_weather, axis=1, weather_bronx=weather_bronx, weather_manhattan=weather_manhattan)

# Combine the UHI index data with the interpolated weather data
combined_data = pd.concat([uhi_data, interpolated_weather_data], axis=1)

# # Save the combined data to a CSV file
# combined_data.to_csv('combined_uhi_weather_data.csv', index=False)

# # Display the first few rows of the combined data
# combined_data.head()


# Define features and target variables for weather prediction
weather_features = ['Latitude', 'Longitude']
weather_targets = ['Air Temp at Surface [degC]', 'Relative Humidity [percent]', 'Avg Wind Speed [m/s]', 'Wind Direction [degrees]', 'Solar Flux [W/m^2]']

# Split the data into training and testing sets for weather prediction
X_weather_train, X_weather_test, y_weather_train, y_weather_test = train_test_split(combined_data[weather_features], combined_data[weather_targets], test_size=0.3, random_state=42)

# Train a Random Forest Regressor for weather prediction
weather_model = RandomForestRegressor(n_estimators=100, random_state=42)
weather_model.fit(X_weather_train, y_weather_train)

# Predict weather data for the test set
y_weather_pred = weather_model.predict(X_weather_test)

# Evaluate the weather prediction model
print("Weather Prediction R-squared:", r2_score(y_weather_test, y_weather_pred))
print("Weather Prediction RMSE:", np.sqrt(mean_squared_error(y_weather_test, y_weather_pred, multioutput='raw_values')))

# Predict weather data for the entire dataset
predicted_weather_data = weather_model.predict(combined_data[weather_features])
predicted_weather_df = pd.DataFrame(predicted_weather_data, columns=weather_targets)

# Combine the predicted weather data with the UHI index data
combined_data_with_predicted_weather = pd.concat([combined_data, predicted_weather_df.add_prefix('Predicted_')], axis=1)

# Define features and target variable for UHI prediction
uhi_features = ['Predicted_Air Temp at Surface [degC]', 'Predicted_Relative Humidity [percent]', 'Predicted_Avg Wind Speed [m/s]', 'Predicted_Wind Direction [degrees]', 'Predicted_Solar Flux [W/m^2]']
uhi_target = 'UHI Index'

# Split the data into training and testing sets for UHI prediction
X_uhi_train, X_uhi_test, y_uhi_train, y_uhi_test = train_test_split(combined_data_with_predicted_weather[uhi_features], combined_data_with_predicted_weather[uhi_target], test_size=0.3, random_state=42)

# Train a Random Forest Regressor for UHI prediction
uhi_model = RandomForestRegressor(n_estimators=100, random_state=42)
uhi_model.fit(X_uhi_train, y_uhi_train)

# Predict UHI index for the test set
y_uhi_pred = uhi_model.predict(X_uhi_test)

# Evaluate the UHI prediction model
print("UHI Prediction R-squared:", r2_score(y_uhi_test, y_uhi_pred))
print("UHI Prediction RMSE:", np.sqrt(mean_squared_error(y_uhi_test, y_uhi_pred)))

# Save the combined data with predicted weather and UHI index to a CSV file
combined_data_with_predicted_weather.to_csv('combined_uhi_weather_data_with_predictions.csv', index=False)

# Display the first few rows of the combined data with predictions
combined_data_with_predicted_weather.head()
