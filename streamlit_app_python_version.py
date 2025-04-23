"""
Python version compatibility script for Streamlit Cloud
This script helps identify the Python environment for our app deployment
"""
import streamlit as st
import sys
import platform

st.set_page_config(page_title="Python Environment Info", page_icon="🐍")

st.title("Python Environment Information")

# Python version
st.header("Python Version")
st.code(sys.version)

# Platform information
st.header("Platform Information")
st.code(platform.platform())

# Package versions
st.header("Key Package Versions")

try:
    import numpy as np
    st.write(f"NumPy: {np.__version__}")
except ImportError:
    st.error("NumPy not installed")

try:
    import pandas as pd
    st.write(f"Pandas: {pd.__version__}")
except ImportError:
    st.error("Pandas not installed")

try:
    import sklearn
    st.write(f"scikit-learn: {sklearn.__version__}")
except ImportError:
    st.error("scikit-learn not installed")

try:
    import matplotlib
    st.write(f"Matplotlib: {matplotlib.__version__}")
except ImportError:
    st.error("Matplotlib not installed")

try:
    import joblib
    st.write(f"joblib: {joblib.__version__}")
except ImportError:
    st.error("joblib not installed")

st.markdown("---")
st.info("This is a utility app for debugging Python environment issues on Streamlit Cloud.")
