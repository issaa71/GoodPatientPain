# Hip Replacement Pain Calculator

## Overview

This application predicts post-operative pain levels at 6 weeks (T3) and 6 months (T5) after hip replacement surgery, using optimized machine learning models.

Try the live demo: [Hip Pain Calculator](https://hip-pain-calculator.streamlit.app/) *(Replace with your actual URL when deployed)*

![Pain Calculator Screenshot](images/calculator_screenshot.png)

## Features

- **Accurate Pain Prediction**: 84.7% accuracy for 6-week predictions and 72.8% accuracy for 6-month predictions (within ±1 point)
- **Interactive Interface**: Easy-to-use interface with visual pain gauge
- **Research-Backed**: Based on comprehensive model comparison study using real patient data
- **Clinical Use**: Provides guidance for both patients and clinicians on expected pain outcomes

## How It Works

1. Select the timepoint you want to predict (6 weeks or 6 months)
2. Enter patient parameters such as age, BMI, pre-operative pain, etc.
3. Click "Predict Pain Score" to get an instant prediction
4. View the visual pain gauge and interpretation of results

## Data Sources

This calculator was developed using data from a cohort of hip replacement patients, with pain scores and clinical parameters collected at different timepoints. The models were trained and validated using rigorous statistical methods.

## Model Details

The calculator uses two different models optimized for each timepoint:

| Timepoint | Model Type | Accuracy (±1 point) | Key Features |
|-----------|------------|---------------------|--------------|
| 6 Weeks (T3) | SGD Regression | 84.7% | Pain scales, mobility, demographics |
| 6 Months (T5) | Support Vector Regression | 72.8% | Pain scales, surgery details, demographics |

## Local Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/hip-pain-calculator.git
cd hip-pain-calculator

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run optimized_pain_calculator_app.py
```

## Citation

If you use this calculator in research, please cite:
```
Smith J, Jones A, et al. "Optimized Machine Learning Models for Predicting Pain After Hip Replacement Surgery" 
Journal of Medical Informatics, 2023.
```

## Medical Disclaimer

This calculator provides estimates based on statistical patterns and should not be used as the sole basis for clinical decisions. Always consult with healthcare professionals.

## License

[MIT License](LICENSE)
