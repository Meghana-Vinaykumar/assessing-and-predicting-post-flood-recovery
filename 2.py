import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from geopy.geocoders import Nominatim

# Step 1: Load the dataset
data = pd.read_csv("flood_data1.csv")
print("Original dataset:\n", data)

# Step 2: Preprocess the data
# Assume 'address' is a column in the dataset
geolocator = Nominatim(user_agent="flood_predictor")

# Geocode addresses to get latitude and longitude
data['location'] = data['address'].apply(geolocator.geocode)
data['latitude'] = data['location'].apply(lambda loc: loc.latitude if loc else None)
data['longitude'] = data['location'].apply(lambda loc: loc.longitude if loc else None)

# Drop rows with missing geolocation data
data = data.dropna(subset=['latitude', 'longitude'])
print("Dataset after geocoding:\n", data)

# Step 3: Feature engineering
# Assume 'precipitation', 'river_level', and 'soil_moisture' are columns in the dataset
features = ['precipitation', 'river_level', 'soil_moisture', 'latitude', 'longitude']
X = data[features]
y = data['flood_occurred']  # Binary target: 0 = No flood, 1 = Flood

# Step 4: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data:\n", X_train)
print("Test data:\n", X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test)
print("Predictions:\n", predictions)

# Step 6: Geolocation integration for predictions
prediction_results = pd.DataFrame(X_test)
prediction_results['prediction'] = predictions
prediction_results['address'] = data.loc[prediction_results.index, 'address']

# Display results
print("Prediction Results:\n", prediction_results)

for idx, row in prediction_results.iterrows():
    status = "will" if row['prediction'] == 1 else "will not"
    print(f"The place at address {row['address']} ({row['latitude']}, {row['longitude']}) {status} experience a flood.")
