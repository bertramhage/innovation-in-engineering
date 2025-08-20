import argparse
import pandas as pd
import joblib
from main import train_model, test_on_prod_data

def run_pipeline(data_path):
    """
    Run the complete pipeline: load data, train model, and test it on production data.
    
    Args:
        data_path (str): Path to the CSV file containing the training data
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    print(f"Loaded {len(data)} samples with {sum(data['label'])} fire events")
    print(f"Training model...")
    
    train_model(data)
    
    print(f"Loading trained model...")
    model = joblib.load('fire_detection_model.joblib')
    
    print(f"Testing model on production data...")
    test_on_prod_data(model)
    
    print("Pipeline completed successfully!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Detection Model Pipeline")
    parser.add_argument('--data_path', type=str, default='data/synthetic_fire_data.csv',
                        help='Path to the CSV file containing training data')
    
    args = parser.parse_args()
    run_pipeline(args.data_path)