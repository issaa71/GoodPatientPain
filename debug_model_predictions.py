"""
Model prediction debugging tool for the hip pain calculator
This script tests different input parameter combinations to see if the model produces varied outputs
"""
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Features used for each timepoint
T3_FEATURES = [
    'LOS', 'BMI_Current', 'WOMACP_5', 'WeightCurrent', 'ICOAPC_3',
    'ICOAPC_1', 'AgePreOp', 'WOMACP_3', 'WalkPain', 'MobilityAidWalker',
    'Pre-Op Pain', 'HeightCurrent', 'ResultsRelief', 'WOMACP_1', 'WOMACP_2'
]

T5_FEATURES = [
    'AgePreOp', 'BMI_Current', 'WeightCurrent', 'HeightCurrent', 'LOS',
    'WOMACP_5', 'ResultsRelief', 'ICOAPC_3', 'Pre-Op Pain', 'WalkPain',
    'Approach', 'HeadSize', 'WOMACP_1', 'WOMACP_3', 'ICOAPC_1'
]

def load_models():
    """Load the models and check if they exist"""
    models_found = False
    t3_model = None
    t5_model = None
    
    # Check different potential locations
    model_locations = [
        './',  # Current directory
        './models/',  # Models subdirectory
        '../',  # Parent directory
        '../models/'  # Parent's models subdirectory
    ]
    
    for location in model_locations:
        t3_path = os.path.join(location, 't3_sgd_regression_optimized.joblib')
        t5_path = os.path.join(location, 't5_svr_optimized.joblib')
        
        if os.path.exists(t3_path) and os.path.exists(t5_path):
            try:
                print(f"Found models in: {location}")
                t3_model = joblib.load(t3_path)
                t5_model = joblib.load(t5_path)
                models_found = True
                break
            except Exception as e:
                print(f"Error loading models from {location}: {str(e)}")
    
    if not models_found:
        print("Could not find model files in any expected location.")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir())
    
    return t3_model, t5_model, models_found

def test_predictions_with_varying_inputs(model, features, timepoint):
    """Test model predictions with different input values"""
    if model is None:
        print(f"No model available for {timepoint}")
        return
    
    print(f"\n===== Testing {timepoint} Model =====")
    
    # Create a baseline patient with average values
    baseline_data = {
        'LOS': 3,
        'BMI_Current': 28.0,
        'WOMACP_5': 2,
        'WeightCurrent': 80,
        'ICOAPC_3': 2, 
        'ICOAPC_1': 2,
        'AgePreOp': 65,
        'WOMACP_3': 2,
        'WalkPain': 5,
        'MobilityAidWalker': 0,
        'Pre-Op Pain': 5,
        'HeightCurrent': 170, 
        'ResultsRelief': 3,
        'WOMACP_1': 2,
        'WOMACP_2': 2,
        'Approach': 1,
        'HeadSize': 32
    }
    
    # Ensure baseline has all required features
    feature_set = set(features)
    for feature in list(baseline_data.keys()):
        if feature not in feature_set:
            del baseline_data[feature]
    
    # Test baseline prediction
    baseline_df = pd.DataFrame([baseline_data])[features]
    try:
        baseline_prediction = model.predict(baseline_df)[0]
        baseline_prediction = np.clip(baseline_prediction, 0, 8)
        print(f"Baseline prediction: {baseline_prediction:.2f}")
    except Exception as e:
        print(f"Error with baseline prediction: {str(e)}")
        return
    
    # Test sensitivity to various input changes
    print("\nSensitivity Analysis (how prediction changes with input):")
    
    # List of features to test sensitivity on
    sensitivity_features = {
        'WalkPain': [0, 3, 7, 10],
        'Pre-Op Pain': [0, 3, 7, 10],
        'WOMACP_5': [0, 1, 3, 4],
        'ICOAPC_3': [0, 1, 3, 4],
        'BMI_Current': [20, 25, 30, 35],
        'AgePreOp': [40, 55, 70, 85]
    }
    
    results = {}
    
    # Test each feature's sensitivity
    for feature, values in sensitivity_features.items():
        if feature not in features:
            continue
            
        print(f"\nTesting sensitivity to {feature}:")
        feature_results = []
        
        for value in values:
            test_data = baseline_data.copy()
            test_data[feature] = value
            test_df = pd.DataFrame([test_data])[features]
            
            try:
                prediction = model.predict(test_df)[0]
                prediction = np.clip(prediction, 0, 8)
                print(f"  {feature} = {value} → Prediction: {prediction:.2f}")
                feature_results.append((value, prediction))
            except Exception as e:
                print(f"  Error with {feature} = {value}: {str(e)}")
        
        results[feature] = feature_results
    
    # Plot sensitivity results
    if results:
        plt.figure(figsize=(12, 8))
        
        for i, (feature, values) in enumerate(results.items()):
            if not values:
                continue
                
            x_vals, y_vals = zip(*values)
            plt.subplot(2, 3, i+1)
            plt.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=8)
            plt.title(f"Effect of {feature} on Pain")
            plt.xlabel(feature)
            plt.ylabel("Predicted Pain")
            plt.grid(True, alpha=0.3)
            
            # Calculate range of predictions
            pred_range = max(y) - min(y) if len(y_vals) > 1 else 0
            plt.annotate(f"Range: {pred_range:.2f}", 
                         xy=(0.05, 0.95), 
                         xycoords='axes fraction',
                         fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{timepoint}_sensitivity_analysis.png")
        print(f"Saved sensitivity plot to {timepoint}_sensitivity_analysis.png")
    
    # Test with extreme values to check model robustness
    print("\nTesting with extreme values:")
    
    # Extreme cases
    extreme_cases = [
        ("High pain indicators", {k: 4 if 'WOMAC' in k or 'ICOA' in k else 
                                  10 if 'Pain' in k else 
                                  v for k, v in baseline_data.items()}),
        
        ("Low pain indicators", {k: 0 if 'WOMAC' in k or 'ICOA' in k else 
                                 0 if 'Pain' in k else 
                                 v for k, v in baseline_data.items()}),
        
        ("Elderly patient", {**baseline_data, 'AgePreOp': 90, 'BMI_Current': 32, 'MobilityAidWalker': 1}),
        
        ("Young active patient", {**baseline_data, 'AgePreOp': 35, 'BMI_Current': 22, 'MobilityAidWalker': 0})
    ]
    
    for label, case_data in extreme_cases:
        # Keep only features relevant to this model
        filtered_data = {k: v for k, v in case_data.items() if k in features}
        case_df = pd.DataFrame([filtered_data])[features]
        
        try:
            prediction = model.predict(case_df)[0]
            prediction = np.clip(prediction, 0, 8)
            print(f"  {label}: Prediction = {prediction:.2f}")
        except Exception as e:
            print(f"  Error with {label}: {str(e)}")

def main():
    # Load models
    t3_model, t5_model, models_found = load_models()
    
    if not models_found:
        print("No models found. Cannot proceed with testing.")
        return
    
    # Test T3 model
    test_predictions_with_varying_inputs(t3_model, T3_FEATURES, "T3")
    
    # Test T5 model
    test_predictions_with_varying_inputs(t5_model, T5_FEATURES, "T5")
    
    print("\nTesting complete.")

if __name__ == "__main__":
    main()
