import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('YOUR_CSV_FILE_PATH HERE')

# Remove thousands separator and fix decimal separator
data['Now'] = data['Now'].str.replace('.', '').str.replace(',', '.').astype(float)

# Prepare the dataset
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
X = data[['Days']]
y = data['Now']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make a prediction
future_date = pd.to_datetime('2023-12-29')
days_future = (future_date - data['Date'].min()).days
price_prediction = model.predict([[days_future]])

print("Price Prediction:", price_prediction[0])
