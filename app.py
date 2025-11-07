import streamlit as st
import pandas as pd
from io import StringIO
import sys

# Simplified REQUIRED_COLS for testing the dashboard structure
try:
    from utils import REQUIRED_COLS
except ImportError:
    st.set_page_config(layout="wide", page_title="Cricket Performance Dashboard")
    st.error("Setup Error: Cannot find 'utils.py'. Please ensure all files are present.")
    sys.exit()

st.set_page_config(layout="wide", page_title="Cricket Performance Dashboard")

# --- GLOBAL DATA HANDLING ---
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = pd.DataFrame()
    
st.title("Cricket Analysis Dashboard (Trial)")

# --- FILE UPLOADER (Located in the sidebar) ---
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file:
    try:
        data = uploaded_file.getvalue().decode("utf-8")
        df_raw = pd.read_csv(StringIO(data))
        
        # Validation: Only check essential columns for this trial
        missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
        
        if missing_cols:
            st.sidebar.error(f"Missing required columns: {', '.join(missing_cols)}. Check your file.")
            st.session_state['df_raw'] = pd.DataFrame()
        else:
            st.session_state['df_raw'] = df_raw
            st.sidebar.success("Data loaded successfully! **The navigation link should now appear.**")

    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.session_state['df_raw'] = pd.DataFrame()
        
# --- LANDING PAGE CONTENT ---
st.header("Welcome! Please upload data in the sidebar.")
if st.session_state['df_raw'].empty:
    st.info("Upload CSV to see navigation links (like 'Batters') in the sidebar.")
else:
    st.success("Data is ready! Please select 'Batters' from the sidebar menu.")
