"""
Debugging version of the hip pain calculator app
This version includes detailed logging of predictions and input values
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import json
import datetime

# App configuration
st.set_page_config(
    page_title="Hip Pain Calculator - Debug Mode",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Hip Replacement Pain Calculator - DEBUG MODE")
st.markdown("""
This is a debugging version of the calculator that shows detailed model information
and prediction details to help diagnose issues.
""")

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

# Feature descriptions for user interface - Updated with specific WOMAC/ICOA numbers
FEATURE_DESCRIPTIONS = {
    'LOS': 'Length of Stay (days)',
    'BMI_Current': 'Body Mass Index',
    'WOMACP_5': 'WOMAC P5 - Walking on flat surface (0-4)',
    'WOMACP_3': 'WOMAC P3 - Night pain in bed (0-4)',
    'WOMACP_1': 'WOMAC P1 - Walking pain (0-4)',
    'WOMACP_2': 'WOMAC P2 - Stairs pain (0-4)',
    'WeightCurrent': 'Weight (kg)',
    'ICOAPC_3': 'ICOA P3 - Walking pain (0-4)',
    'ICOAPC_1': 'ICOA P1 - Night pain (0-4)',
    'AgePreOp': 'Age at Surgery',
    'WalkPain': 'Walking Pain (0-10)',
    'MobilityAidWalker': 'Using Walking Aid (1=Yes, 0=No)',
    'Pre-Op Pain': 'Pre-op Pain Level (0-10)',
    'HeightCurrent': 'Height (cm)',
    'ResultsRelief': 'Expected Results - Pain Relief (1-5)',
    'Approach': 'Surgical Approach (Anterior=1, Posterior=0)',
    'HeadSize': 'Femoral Head Size (mm)',
}

# Debug logging function
def log_debug_info(message, data=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    if data:
        if isinstance(data, (dict, list)):
            log_entry += f"\n{json.dumps(data, indent=2)}"
        else:
            log_entry += f"\n{data}"
    
    st.session_state.debug_log.append(log_entry)

# Helper function for loading models
def load_models():
    """Load the prediction models - with detailed logging"""
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = []
    
    log_debug_info("Attempting to load models")
    
    # Log the current working directory and contents
    cwd = os.getcwd()
    log_debug_info(f"Current working directory: {cwd}")
    
    try:
        dir_contents = os.listdir()
        log_debug_info(f"Directory contents: {dir_contents}")
    except Exception as e:
        log_debug_info(f"Error listing directory: {str(e)}")
    
    # Check for models in all possible locations
    model_locations = [
        './',  # Current directory
        './models/',  # Models subdirectory
        '../',  # Parent directory
        '../models/'  # Parent's models subdirectory
    ]
    
    for location in model_locations:
        t3_path = os.path.join(location, 't3_sgd_regression_optimized.joblib')
        t5_path = os.path.join(location, 't5_svr_optimized.joblib')
        
        log_debug_info(f"Checking for models in: {location}")
        
        if os.path.exists(t3_path) and os.path.exists(t5_path):
            try:
                log_debug_info(f"Found model files. Loading from {location}")
                t3_model = joblib.load(t3_path)
                t5_model = joblib.load(t5_path)
                
                # Log model info
                log_debug_info("T3 model type:", type(t3_model).__name__)
                log_debug_info("T5 model type:", type(t5_model).__name__)
                
                return t3_model, t5_model, True
            except Exception as e:
                log_debug_info(f"Error loading models from {location}: {str(e)}")
    
    log_debug_info("No model files found. Running in demo mode.")
    return None, None, False

# Function for making predictions
def predict_pain(patient_data, timepoint='T3'):
    """
    Predict pain score with detailed logging
    """
    log_debug_info(f"Making prediction for {timepoint} with patient data:", patient_data)
    
    # Make sure models are loaded
    t3_model, t5_model, models_loaded = load_models()
    
    # Log model loading status
    log_debug_info(f"Models loaded successfully: {models_loaded}")
    
    if not models_loaded:
        log_debug_info("Using demo mode for prediction")
        return {
            'error': True,
            'message': "Models could not be loaded. Running in demo mode."
        }
    
    # Select the appropriate model and features
    if timepoint == 'T3':
        model = t3_model
        required_features = T3_FEATURES
        buffer_accuracy = 0.847  # From optimized_models_results.csv
    else:  # T5
        model = t5_model
        required_features = T5_FEATURES
        buffer_accuracy = 0.728  # From optimized_models_results.csv
    
    log_debug_info(f"Selected model for {timepoint} with features:", required_features)
    
    # Check if all required features are provided
    missing_features = [f for f in required_features if f not in patient_data]
    if missing_features:
        log_debug_info(f"Missing features detected:", missing_features)
        return {
            'error': True,
            'message': f"Missing required features: {', '.join(missing_features)}"
        }
    
    # Create a DataFrame with the patient data
    df = pd.DataFrame([patient_data])
    
    # Select only the required features in the correct order
    log_debug_info("Original dataframe columns:", list(df.columns))
    df = df[required_features]
    log_debug_info("Filtered dataframe columns:", list(df.columns))
    
    # Log the dataframe values
    log_debug_info("Input data for prediction:", df.to_dict('records')[0])
    
    # Make prediction
    try:
        # Log the model steps if possible
        if hasattr(model, 'steps'):
            log_debug_info("Model pipeline steps:", [step[0] for step in model.steps])
        
        # The model pipeline handles preprocessing
        raw_prediction = model.predict(df)[0]
        log_debug_info(f"Raw prediction value: {raw_prediction}")
        
        # Clip to valid range
        pain_score = np.clip(raw_prediction, 0, 8)
        log_debug_info(f"Clipped prediction value: {pain_score}")
        
        # Round to nearest 0.1
        pain_score = round(pain_score * 10) / 10
        log_debug_info(f"Final rounded prediction: {pain_score}")
        
        # Interpret pain level
        if pain_score <= 2:
            interpretation = "Minimal pain"
            color = "green"
        elif pain_score <= 4:
            interpretation = "Mild pain"
            color = "yellowgreen"
        elif pain_score <= 6:
            interpretation = "Moderate pain"
            color = "orange"
        else:
            interpretation = "Severe pain"
            color = "red"
        
        log_debug_info(f"Pain interpretation: {interpretation}")
        
        return {
            'error': False,
            'pain_score': pain_score,
            'raw_prediction': raw_prediction,
            'confidence': buffer_accuracy,
            'interpretation': interpretation,
            'color': color,
            'timepoint': timepoint
        }
    except Exception as e:
        log_debug_info(f"Prediction error: {str(e)}")
        import traceback
        log_debug_info("Traceback:", traceback.format_exc())
        return {
            'error': True,
            'message': f"Prediction error: {str(e)}"
        }

# Demo function for when models aren't available
def demo_prediction(timepoint, use_variety=True):
    """Return a demo prediction with variety if requested"""
    log_debug_info(f"Using demo prediction for {timepoint}, variety: {use_variety}")
    
    # In demo mode, generate slightly different predictions for different timepoints
    if timepoint == 'T3':
        # Generate values between 0.8 and 1.2 to show some variety if requested
        pain_score = 1.0 if not use_variety else np.random.uniform(0.8, 3.5)
        confidence = 0.847
    else:
        # Generate values between 2.5 and 3.5 for T5
        pain_score = 3.0 if not use_variety else np.random.uniform(2.5, 4.5)
        confidence = 0.728
    
    # Round to nearest 0.1
    pain_score = round(pain_score * 10) / 10
    log_debug_info(f"Demo mode pain score: {pain_score}")
    
    # Interpret pain level
    if pain_score <= 2:
        interpretation = "Minimal pain"
        color = "green"
    elif pain_score <= 4:
        interpretation = "Mild pain"
        color = "yellowgreen"
    elif pain_score <= 6:
        interpretation = "Moderate pain"
        color = "orange"
    else:
        interpretation = "Severe pain"
        color = "red"
    
    return {
        'error': False,
        'pain_score': pain_score,
        'confidence': confidence,
        'interpretation': interpretation,
        'color': color,
        'timepoint': timepoint,
        'demo': True
    }

# Function for creating the gauge chart
def create_gauge_chart(value, min_value=0, max_value=8, colors=None, title=None):
    """Create a gauge chart for the pain score"""
    if colors is None:
        colors = {
            'green': (0, 2),
            'yellowgreen': (2, 4),
            'orange': (4, 6),
            'red': (6, 8)
        }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
    
    # Start and end angles in radians
    min_theta = np.pi/2
    max_theta = np.pi*3/2
    
    # Convert pain score to angle
    value_theta = min_theta + (max_theta - min_theta) * (value - min_value) / (max_value - min_value)
    
    # Draw colored arcs for each pain level
    for color, (low, high) in colors.items():
        start_theta = min_theta + (max_theta - min_theta) * (low - min_value) / (max_value - min_value)
        end_theta = min_theta + (max_theta - min_theta) * (high - min_value) / (max_value - min_value)
        
        # Draw the colored arc
        ax.add_patch(Wedge((0, 0), 0.9, 
                           np.degrees(start_theta), 
                           np.degrees(end_theta),
                           width=0.4, 
                           facecolor=color, 
                           alpha=0.8))
    
    # Draw pointer line
    ax.plot([0, 0.7 * np.cos(value_theta)], [0, 0.7 * np.sin(value_theta)], 'k-', lw=3)
    
    # Draw center circle
    ax.add_patch(plt.Circle((0, 0), 0.1, facecolor='dimgray', edgecolor='none'))
    
    # Add text labels
    for i, label in enumerate(['0', '2', '4', '6', '8']):
        angle = min_theta + (max_theta - min_theta) * i / 4
        ax.text(0.7 * np.cos(angle), 0.7 * np.sin(angle), label, 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add interpretation labels
    for color, (low, high) in colors.items():
        mid_value = (low + high) / 2
        mid_theta = min_theta + (max_theta - min_theta) * (mid_value - min_value) / (max_value - min_value)
        
        # Determine text color
        if color in ['green', 'yellowgreen']:
            text_color = 'darkgreen'
        else:
            text_color = 'darkred'
        
        # Get pain level name
        if color == 'green':
            level = 'Minimal'
        elif color == 'yellowgreen':
            level = 'Mild'
        elif color == 'orange':
            level = 'Moderate'
        else:
            level = 'Severe'
        
        # Add text
        ax.text(0.5 * np.cos(mid_theta), 0.5 * np.sin(mid_theta), level, 
                ha='center', va='center', fontsize=10, color=text_color, fontweight='bold')
    
    # Add title if provided
    if title:
        plt.title(title, pad=20, fontsize=14, fontweight='bold')
    
    # Hide axis elements
    ax.set_rmax(1)
    ax.set_rticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    return fig

# Main app layout
def app():
    # Initialize debug log if not exists
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = []
    
    log_debug_info("App initialized")
    
    # Load models
    t3_model, t5_model, models_loaded = load_models()
    
    # Display warning if models not loaded
    if not models_loaded:
        st.warning("⚠️ Models could not be loaded. Running in demo mode.")
        log_debug_info("Warning displayed: Models not loaded, running in demo mode")
    
    # Add debug toggles in sidebar
    st.sidebar.title("Debug Options")
    show_debug_log = st.sidebar.checkbox("Show Debug Log", value=True)
    use_demo_variety = st.sidebar.checkbox("Use Variety in Demo Mode", value=True)
    force_demo_mode = st.sidebar.checkbox("Force Demo Mode", value=False)
    
    # Sidebar - Select timepoint
    st.sidebar.title("Settings")
    timepoint = st.sidebar.radio(
        "Select Prediction Timepoint",
        ["T3 (3 years)", "T5 (5 years)"]
    )
    
    # Get the appropriate feature list
    if timepoint == "T3 (3 years)":
        feature_list = T3_FEATURES
        prediction_timepoint = 'T3'
    else:
        feature_list = T5_FEATURES
        prediction_timepoint = 'T5'
    
    log_debug_info(f"Selected timepoint: {timepoint} with feature list of {len(feature_list)} items")
    
    # Create main columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("Patient Parameters")
        
        # Create input form
        with st.form("patient_form"):
            # Create multiple columns for inputs
            form_cols = st.columns(2)
            
            # Dictionary to store patient data
            patient_data = {}
            
            # Display inputs for each feature
            for i, feature in enumerate(feature_list):
                col_idx = i % 2  # Alternate between columns
                
                # Get the description or default to the feature name
                description = FEATURE_DESCRIPTIONS.get(feature, feature)
                
                # Determine the appropriate input widget
                if feature == 'MobilityAidWalker':
                    patient_data[feature] = form_cols[col_idx].selectbox(
                        description,
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No"
                    )
                elif feature == 'Approach':
                    patient_data[feature] = form_cols[col_idx].selectbox(
                        description,
                        options=[0, 1],
                        format_func=lambda x: "Anterior" if x == 1 else "Posterior"
                    )
                elif 'WOMAC' in feature or 'ICOA' in feature:
                    # WOMAC and ICOA scores are 0-4
                    patient_data[feature] = form_cols[col_idx].slider(
                        description,
                        min_value=0,
                        max_value=4,
                        value=2
                    )
                elif feature == 'ResultsRelief':
                    # ResultsRelief is 1-5
                    patient_data[feature] = form_cols[col_idx].slider(
                        description,
                        min_value=1,
                        max_value=5,
                        value=3
                    )
                elif feature in ['WalkPain', 'Pre-Op Pain']:
                    # Pain scores are 0-10
                    patient_data[feature] = form_cols[col_idx].slider(
                        description,
                        min_value=0,
                        max_value=10,
                        value=5
                    )
                elif feature == 'LOS':
                    patient_data[feature] = form_cols[col_idx].number_input(
                        description,
                        min_value=1,
                        max_value=14,
                        value=3
                    )
                elif feature == 'HeadSize':
                    patient_data[feature] = form_cols[col_idx].selectbox(
                        description,
                        options=[28, 32, 36, 40],
                        index=1
                    )
                elif feature == 'AgePreOp':
                    patient_data[feature] = form_cols[col_idx].number_input(
                        description,
                        min_value=30,
                        max_value=95,
                        value=65
                    )
                elif feature == 'HeightCurrent':
                    patient_data[feature] = form_cols[col_idx].number_input(
                        description,
                        min_value=140,
                        max_value=210,
                        value=170
                    )
                elif feature == 'WeightCurrent':
                    patient_data[feature] = form_cols[col_idx].number_input(
                        description,
                        min_value=40,
                        max_value=180,
                        value=75
                    )
                elif feature == 'BMI_Current':
                    # Auto-calculate BMI if possible
                    if 'WeightCurrent' in patient_data and 'HeightCurrent' in patient_data:
                        weight = patient_data['WeightCurrent']
                        height = patient_data['HeightCurrent'] / 100  # Convert to meters
                        bmi = weight / (height * height)
                        patient_data[feature] = form_cols[col_idx].number_input(
                            description,
                            min_value=15.0,
                            max_value=50.0,
                            value=bmi
                        )
                    else:
                        patient_data[feature] = form_cols[col_idx].number_input(
                            description,
                            min_value=15.0,
                            max_value=50.0,
                            value=25.0
                        )
                else:
                    # Default to a number input
                    patient_data[feature] = form_cols[col_idx].number_input(
                        description,
                        value=0
                    )
            
            # Auto-calculate BMI if Weight and Height are changed
            if 'WeightCurrent' in patient_data and 'HeightCurrent' in patient_data and 'BMI_Current' in patient_data:
                weight = patient_data['WeightCurrent']
                height = patient_data['HeightCurrent'] / 100  # Convert to meters
                bmi = weight / (height * height)
                patient_data['BMI_Current'] = round(bmi, 1)
            
            # Submit button
            submitted = st.form_submit_button("Predict Pain Score")
    
    # Display prediction result
    with col2:
        st.header("Prediction Result")
        
        if submitted:
            log_debug_info("Prediction requested with form submission")
            
            # Make prediction (use demo mode if forced or if models not loaded)
            if not models_loaded or force_demo_mode:
                log_debug_info("Using demo mode (forced or models not available)")
                result = demo_prediction(prediction_timepoint, use_demo_variety)
            else:
                log_debug_info("Using actual model prediction")
                result = predict_pain(patient_data, prediction_timepoint)
            
            if result['error']:
                st.error(result['message'])
                log_debug_info(f"Error displayed: {result['message']}")
            else:
                # Display the prediction
                pain_score = result['pain_score']
                log_debug_info(f"Displaying prediction: {pain_score}")
                
                # Display raw prediction if available
                if 'raw_prediction' in result:
                    st.write(f"Raw model output: {result['raw_prediction']:.4f}")
                
                # Display the gauge
                gauge_chart = create_gauge_chart(
                    pain_score,
                    title=f"Predicted {timepoint} Pain Score: {pain_score:.1f}"
                )
                st.pyplot(gauge_chart)
                
                # Display interpretation
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: {result['color']}80;'>"
                    f"<h3 style='text-align: center; color: {'black' if result['color'] in ['green', 'yellowgreen'] else 'white'};'>"
                    f"{result['interpretation']}</h3>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                # Display confidence
                st.info(f"Prediction Confidence: {result['confidence']*100:.1f}% within ±1 point")
                
                # Show demo mode notice if necessary
                if result.get('demo', False):
                    st.warning("⚠️ Running in demo mode - this is a sample prediction")
        else:
            st.info("Enter patient information and click 'Predict Pain Score' to get a prediction.")
    
    # Display debug log if enabled
    if show_debug_log:
        st.markdown("---")
        st.subheader("Debug Log")
        
        # Create expandable section for each log entry
        for i, log_entry in enumerate(st.session_state.debug_log):
            with st.expander(f"Log #{i+1}"):
                st.code(log_entry)
    
    # Add information about the model performance
    st.markdown("---")
    st.subheader("About the Model")
    
    if timepoint == "T3 (3 years)":
        st.markdown("""
        **T3 (3 years) Model Information:**
        - Model type: Optimized SGD Regression
        - Accuracy: 84.7% of predictions within ±1 point of actual pain score
        - Features: Using 15 most important clinical features
        """)
    else:
        st.markdown("""
        **T5 (5 years) Model Information:**
        - Model type: Optimized Support Vector Regression (SVR)
        - Accuracy: 72.8% of predictions within ±1 point of actual pain score
        - Features: Using 15 most important clinical features
        """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    **Medical Disclaimer:** This calculator provides estimates based on statistical patterns and should not be used as the sole basis for clinical decisions. Always consult with healthcare professionals.
    """)
    
    # Repository link
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Source Code:** [GitHub Repository](https://github.com/your-username/hip-pain-calculator)  
    **Report Issues:** [GitHub Issues](https://github.com/your-username/hip-pain-calculator/issues)
    """)
    
    # Version info
    st.sidebar.markdown(f"**Version:** DEBUG-1.0.2")

if __name__ == "__main__":
    app()
