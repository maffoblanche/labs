import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import requests

# Step 1: Data Preprocessing
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def explore_data(data):
    print("Data Head:\n", data.head())
    print("\nData Info:\n", data.info())
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nData Description:\n", data.describe())

def preprocess_data(data):
    # Ensure data is sorted chronologically
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.sort_values(by='timestamp', inplace=True)
    
    # Convert time-based features
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    
    # Handle missing values
    data.ffill(inplace=True)
    
    return data

# Step 2: Feature Engineering
def create_time_features(data):
    # Extract additional time-based features
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    return data

def create_lag_features(data, lag=1):
    for i in range(1, lag + 1):
        data[f'traffic_flow_lag_{i}'] = data['traffic_flow'].shift(i)
    data.dropna(inplace=True)
    return data

def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Step 3: Data Splitting
def split_data(data):
    X = data.drop(['traffic_flow', 'timestamp'], axis=1)
    y = data['traffic_flow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Model Building
def build_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

# Step 5: Model Training and Evaluation
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

def plot_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()

# Additional Challenge: Real-Time Predictions
def fetch_live_data(api_url):
    response = requests.get(api_url)
    live_data = response.json()
    return live_data

def preprocess_live_data(live_data):
    print("Live Data:", live_data)  # Print live data to inspect its structure
    # Extract relevant fields from live data
    live_df = pd.DataFrame([{
        'timestamp': live_data['dt'],
        'temperature': live_data['main']['temp'],
        'humidity': live_data['main']['humidity'],
        'weather': live_data['weather'][0]['main']
    }])
    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='s')  # Convert Unix timestamp
    live_df['hour'] = live_df['timestamp'].dt.hour
    live_df['day_of_week'] = live_df['timestamp'].dt.dayofweek
    live_df = create_time_features(live_df)
    live_df = create_lag_features(live_df, lag=3)
    return live_df

if __name__ == "__main__":
        # Define the file path correctly
    file_path = r'C:\Users\blanche\Desktop\B_tech\AI\Lab_5_Traffic_Flow_Prediction_Using_Neural\synthetic_traffic_data - synthetic_traffic_data.csv'
    data = load_data(file_path)
    explore_data(data)
    data = preprocess_data(data)
    explore_data(data)  # Check data after preprocessing


    
    # Step 2: Feature Engineering
    data = create_time_features(data)
    data = create_lag_features(data, lag=3)
    
    # Step 3: Data Splitting
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Step 4: Normalization/Standardization
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Step 5: Model Building
    model = build_model()
    
    # Step 6: Model Training and Evaluation
    train_model(model, X_train_scaled, y_train)
    mse, predictions = evaluate_model(model, X_test_scaled, y_test)
    print(f'Mean Squared Error: {mse}')
    
    # Step 7: Visualization
    plot_results(y_test, predictions)
    
    # Additional Challenge: Real-Time Predictions
    API_KEY = "e170a575a38552514eac01bd7af8d42c"
    city = "Yaounde"
    api_url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
    live_data = fetch_live_data(api_url)
    live_df = preprocess_live_data(live_data)
    live_X = live_df.drop(['traffic_flow', 'timestamp'], axis=1, errors='ignore')  # Avoid KeyError
    live_X_scaled = scaler.transform(live_X)
    live_predictions = model.predict(live_X_scaled)
    print("Real-Time Predictions:", live_predictions)