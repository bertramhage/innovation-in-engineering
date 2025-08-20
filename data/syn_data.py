import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

def generate_normal_window(start_time, window_size=30, temp_base=26, hum_base=53):
    """Generate a window of normal temperature and humidity readings"""
    timestamps = [start_time + timedelta(minutes=i) for i in range(window_size)]
    
    # Normal patterns with some small random fluctuations
    temps = temp_base + np.random.normal(0, 0.5, window_size)
    hums = hum_base + np.random.normal(0, 0.8, window_size)
    
    return timestamps, temps, hums, 0  # Label 0 = no fire

def generate_fire_window(start_time, window_size=30, temp_base=26, hum_base=53):
    """Generate a window of temperature and humidity readings with fire pattern"""
    timestamps = [start_time + timedelta(minutes=i) for i in range(window_size)]
    
    # Start with normal readings
    temps = temp_base + np.random.normal(0, 0.5, window_size)
    hums = hum_base + np.random.normal(0, 0.8, window_size)
    
    # Fire can start anywhere, ensuring at least 3 timesteps for development
    min_development_time = 3
    fire_start = np.random.randint(0, window_size - min_development_time)
    
    # Randomize development rates for each fire
    temp_rate = np.random.uniform(0.8, 2.2)  # Rate of temperature increase
    hum_rate = np.random.uniform(0.5, 1.5)   # Rate of humidity decrease
    
    # Create temperature spike and humidity drop
    for i in range(fire_start, window_size):
        # Progressive increase in temperature with randomized rate
        temp_increase = (i - fire_start + 1) * temp_rate
        temps[i] = temp_base + temp_increase + np.random.normal(0, 0.3)
        
        # Progressive decrease in humidity with randomized rate
        hum_decrease = (i - fire_start + 1) * hum_rate
        hums[i] = max(hum_base - hum_decrease + np.random.normal(0, 0.3), hum_base - 10)
    
    return timestamps, temps, hums, 1  # Label 1 = fire

def extract_features(timestamps, temps, hums):
    """Extract features from a window similar to the pipeline"""
    temp_mean = np.mean(temps)
    temp_std = np.std(temps)
    temp_range = np.max(temps) - np.min(temps)
    
    hum_mean = np.mean(hums)
    hum_std = np.std(hums)
    hum_range = np.max(hums) - np.min(hums)
    
    temp_diffs = np.diff(temps)
    hum_diffs = np.diff(hums)
    
    features = {
        'temp_mean': temp_mean,
        'temp_std': temp_std,
        'temp_range': temp_range,
        'hum_mean': hum_mean,
        'hum_std': hum_std,
        'hum_range': hum_range,
        'window_size': len(temps),
    }
    
    # Add temperature and humidity differences - maximum of 29 differences for compatibility
    max_diffs = 29  # Maximum possible in a 30-length window
    
    for i in range(min(len(temp_diffs), max_diffs)):
        features[f'temp_{i}'] = temp_diffs[i]
        features[f'hum_{i}'] = hum_diffs[i]
    
    # For shorter windows, pad with zeros to maintain consistent feature dimensions
    for i in range(len(temp_diffs), max_diffs):
        features[f'temp_{i}'] = 0.0
        features[f'hum_{i}'] = 0.0
    
    return features

def generate_dataset(n_samples=100, min_window_size=10, max_window_size=30, fire_ratio=0.3):
    """Generate a dataset with n_samples windows of varying sizes"""
    all_features = []
    labels = []
    windows = []
    
    start_time = datetime.now()
    
    for i in range(n_samples):
        # Randomly select window size for this sample
        window_size = np.random.randint(min_window_size, max_window_size + 1)
        
        # Decide if this window will contain a fire
        is_fire = np.random.random() < fire_ratio
        
        window_start = start_time + timedelta(minutes=i * max_window_size)
        
        if is_fire:
            timestamps, temps, hums, label = generate_fire_window(window_start, window_size)
        else:
            timestamps, temps, hums, label = generate_normal_window(window_start, window_size)
        
        # Extract features
        features = extract_features(timestamps, temps, hums)
        all_features.append(features)
        labels.append(label)
        
        # Store raw window data for visualization
        windows.append({
            'timestamps': timestamps,
            'temps': temps,
            'hums': hums,
            'label': label,
            'window_size': window_size
        })
    
    # Create DataFrame with all features
    df = pd.DataFrame(all_features)
    df['label'] = labels
    
    return df, windows

def visualize_samples(windows, n_samples=3):
    """Visualize a few sample windows"""
    fire_windows = [w for w in windows if w['label'] == 1]
    normal_windows = [w for w in windows if w['label'] == 0]
    
    fig, axs = plt.subplots(n_samples, 2, figsize=(14, 4 * n_samples))
    
    for i in range(n_samples):
        # Plot fire window
        if i < len(fire_windows):
            fire_window = fire_windows[i]
            axs[i, 0].plot(range(len(fire_window['temps'])), fire_window['temps'], 'r-o', label='Temperature')
            axs[i, 0].set_title(f'Fire Window {i+1} (size: {fire_window["window_size"]})')
            axs[i, 0].set_ylabel('Temperature (°C)')
            axs[i, 0].legend(loc='upper left')
            
            ax2 = axs[i, 0].twinx()
            ax2.plot(range(len(fire_window['hums'])), fire_window['hums'], 'b-o', label='Humidity')
            ax2.set_ylabel('Humidity (%)')
            ax2.legend(loc='upper right')
        
        # Plot normal window
        if i < len(normal_windows):
            normal_window = normal_windows[i]
            axs[i, 1].plot(range(len(normal_window['temps'])), normal_window['temps'], 'r-o', label='Temperature')
            axs[i, 1].set_title(f'Normal Window {i+1} (size: {normal_window["window_size"]})')
            axs[i, 1].set_ylabel('Temperature (°C)')
            axs[i, 1].legend(loc='upper left')
            
            ax2 = axs[i, 1].twinx()
            ax2.plot(range(len(normal_window['hums'])), normal_window['hums'], 'b-o', label='Humidity')
            ax2.set_ylabel('Humidity (%)')
            ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('synthetic_data_samples.png')
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate synthetic fire detection data.")
    parser.add_argument('--test_size', type=int, default=100, help='Number of windows to generate (n_samples)')
    parser.add_argument('--fire_ratio', type=float, default=0.3, help='Proportion of windows containing fire (0-1)')
    parser.add_argument('--min_window', type=int, default=5, help='Minimum window size')
    parser.add_argument('--max_window', type=int, default=30, help='Maximum window size')
    args = parser.parse_args()

    np.random.seed(42)  # For reproducibility

    # Generate dataset with user-specified number of windows and fire ratio
    df, windows = generate_dataset(
        n_samples=args.test_size, 
        min_window_size=args.min_window, 
        max_window_size=args.max_window, 
        fire_ratio=args.fire_ratio
    )

    # Save to CSV
    df.to_csv('data/synthetic_fire_data.csv', index=False)
    
    # Print statistics
    print(f"Generated {len(df)} windows of data")
    print(f"Fire windows: {sum(df['label'])}")

    # Visualize some samples
    visualize_samples(windows, n_samples=3)