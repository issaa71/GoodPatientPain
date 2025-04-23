# GitHub Repository Setup Checklist for Streamlit Deployment

Use this checklist to ensure you've included all necessary files for deploying your Hip Pain Calculator to Streamlit Cloud.

## Essential Files

- [ ] **`optimized_pain_calculator_app.py`** - The main Streamlit application
- [ ] **`requirements.txt`** - Dependencies list for Streamlit Cloud
- [ ] **`README.md`** - Repository documentation
- [ ] **`LICENSE`** - MIT License

## Model Files

- [ ] **`t3_sgd_regression_optimized.joblib`** - T3 (6 weeks) prediction model
- [ ] **`t5_svr_optimized.joblib`** - T5 (6 months) prediction model

## Configuration Files

- [ ] **`.streamlit/config.toml`** - Streamlit configuration for theming
- [ ] **`.gitignore`** - Ignore unnecessary files in Git

## Additional Documentation (Optional but Recommended)

- [ ] **`COMPARISON_TABLES.md`** - Model performance tables
- [ ] **`MODEL_COMPARISON_REPORT.md`** - Detailed model analysis
- [ ] **`images/calculator_screenshot.png`** - Screenshot for README

## Directory Structure

Your repository should have a structure like this:

```
hip-pain-calculator/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── LICENSE
├── README.md
├── COMPARISON_TABLES.md (optional)
├── MODEL_COMPARISON_REPORT.md (optional)
├── images/ (optional)
│   └── calculator_screenshot.png
├── optimized_pain_calculator_app.py
├── requirements.txt
├── t3_sgd_regression_optimized.joblib
└── t5_svr_optimized.joblib
```

## Repository Settings

- [ ] **Public repository** - Required for Streamlit Community Cloud
- [ ] **Main branch** configured as default
- [ ] **About section** filled with relevant information
- [ ] **Topics** added (e.g., streamlit, machine-learning, healthcare)

## Streamlit Cloud Setup

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Login with your GitHub account
3. Click "New app"
4. Select your repository
5. Set Main file path to: `optimized_pain_calculator_app.py`
6. Deploy!

## Git LFS Notice

If your model files are large (>100MB), you'll need to use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.joblib"

# Add .gitattributes
git add .gitattributes

# Commit and push as usual
git add .
git commit -m "Initial commit with LFS tracking for model files"
git push
