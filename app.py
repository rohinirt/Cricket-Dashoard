import streamlit as st
import pandas as pd
from io import StringIO
import sys

# Attempt to import REQUIRED_COLS from utils.py for data validation.
try:
    from utils import REQUIRED_COLS
except ImportError:
    # Exit gracefully if utils.py is missing, as it contains core logic.
    st.set_page_config(layout="wide", page_title="Cricket Performance Dashboard")
    st.error("Setup Error: Cannot find 'utils.py'. Please ensure 'app.py' and 'utils.py' are in the root directory.")
    sys.exit()

st.set_page_config(layout="wide", page_title="Cricket Performance Dashboard")

# --- GLOBAL DATA HANDLING ---

# Initialize session state for data sharing across pages
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = pd.DataFrame()
    
st.title("Cricket Analysis Dashboard")

# --- FILE UPLOADER (Located in the sidebar) ---
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file:
    try:
        data = uploaded_file.getvalue().decode("utf-8")
        df_raw = pd.read_csv(StringIO(data))
        
        # Validation: Check for required columns
        missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
        if missing_cols:
            st.sidebar.error(f"Missing required columns: {', '.join(missing_cols)}. Please check your file.")
            st.session_state['df_raw'] = pd.DataFrame() # Clear session on failure
        else:
            # Store the data for all pages to access
            st.session_state['df_raw'] = df_raw
            st.sidebar.success("Data loaded successfully! Select a page to view analysis.")

    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.session_state['df_raw'] = pd.DataFrame()
        
# --- LANDING PAGE CONTENT ---

st.header("Welcome! Please Select an Analysis Page.")
st.markdown("---")

if st.session_state['df_raw'].empty:
    st.info("""
        **To begin your detailed analysis:**
        1. Upload your CSV data file using the uploader in the **sidebar**.
        2. Once uploaded, the **navigation links** (e.g., 'Batters') will appear in the sidebar.
        3. Click on the desired page to view the charts.
    """)
else:
    st.success("""
        **Data is Ready!**
        Please select one of the analysis pages from the **sidebar menu** to view the dashboard.
    """)
