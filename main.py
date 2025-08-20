import numpy as np
import pandas as pd
import requests
import json
import time
import joblib
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report

API_KEY = os.getenv('API_KEY', None)
if not API_KEY:
    raise ValueError("API_KEY is not set.")
CHANNEL_ID = 3039257
THINGSPEAK_URL = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json'

def get_window(window_size = 30, start_time: str | None = None, end_time: str | None = None) -> list[dict]:
    params = { 'api_key': API_KEY, 'results': window_size }
    if end_time:
        params['end'] = end_time
    if start_time:
        params['start'] = start_time
    r = requests.get(THINGSPEAK_URL, params=params, timeout=5)
    r.raise_for_status()
    data = r.json()
    return data['feeds']

def feature_extraction(data: list[dict]) -> np.ndarray:

    n = len(data)

    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['temp'] = pd.to_numeric(df['field1'], errors='coerce')
    df['hum'] = pd.to_numeric(df['field2'], errors='coerce')

    temp_mean = df['temp'].mean()
    temp_std = df['temp'].std()
    temp_range = df['temp'].max() - df['temp'].min()
    hum_mean = df['hum'].mean()
    hum_std = df['hum'].std()
    hum_range = df['hum'].max() - df['hum'].min()
    window_size = len(df)
    temp_features = np.diff(df['temp'].values[:n+1])
    hum_features = np.diff(df['hum'].values[:n+1])

    # If the number of features is less than expected, pad with zeros
    if len(temp_features) < 29:
        temp_features = np.pad(temp_features, (0, 29 - len(temp_features)), 'constant')
    if len(hum_features) < 29:
        hum_features = np.pad(hum_features, (0, 29 - len(hum_features)), 'constant')

    features = np.concatenate([
        [temp_mean, temp_std, temp_range, hum_mean, hum_std, hum_range, window_size],
        temp_features, hum_features
    ])

    return features.reshape(1, -1)

# columns: 'temp_mean','temp_std','temp_range','hum_mean','hum_std','hum_range',
#          'temp_0'...'temp_29','hum_0'...'hum_29','label'

def train_model(data: pd.DataFrame):
        
    X = data.drop('label', axis=1).values
    y = data['label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Calculate class weight to handle imbalance
    class_weight = {
        0: 1.0,
        1: (y_train==0).sum()/(y_train==1).sum()
    }
    
    clf = DecisionTreeClassifier(
        class_weight=class_weight,
        random_state=42,
        min_samples_split=10,
        min_samples_leaf=4,
        max_depth=3,
        ccp_alpha=0.02  # Pruning parameter
    )

    param_grid = {
        'max_depth': [2, 3],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [4, 8],
        'ccp_alpha': [0.01, 0.02, 0.05]
    }
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', n_jobs=4)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    probs = best.predict_proba(X_val)[:,1]
    print('Val AUC:', roc_auc_score(y_val, probs))

    T = 0.5 # Threshold
    preds = (probs >= T).astype(int)
    print(classification_report(y_val, preds))

    # Save the model
    joblib.dump(best, 'fire_detection_model.joblib')

def test_on_prod_data(model):
    """Test the model on production data"""
    # Test 1 - No fire
    with open('test_data/no_fire_test_window.json', 'r') as f:
        data = json.load(f)
    X = feature_extraction(data)
    prob = model.predict_proba(X)[0, 1]
    print(f'No fire test probability: {prob:.4f}')

    # Test 2 - Yes fire
    with open('test_data/fire_test_window.json', 'r') as f:
        data = json.load(f)
    X = feature_extraction(data)
    prob = model.predict_proba(X)[0, 1]
    print(f'Fire test probability: {prob:.4f}')

def run(threshold: float = 0.5):
    """
    Start live detection
    """

    model = joblib.load('fire_detection_model.joblib')

    print("Starting live fire detection...")
    while True:
        start_time = (time.gmtime(time.time() - 600))
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", start_time)
        data = get_window(window_size=30, start_time=start_time) # Get latest 30 readings
        if not data:
            print("No fire detected in the last 10 minutes.")
            time.sleep(20)
            continue
        X = feature_extraction(data)
        prob = model.predict_proba(X)[0, 1]
        if prob >= threshold:
            print(f"FIRE DETECTED!")
        else:
            print(f"No fire detected, probability: {prob:.4f}")
        time.sleep(20)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fire Detection Model")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for fire detection probability')
    
    args = parser.parse_args()
    run(args.threshold)