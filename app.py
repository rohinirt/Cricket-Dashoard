# app.py - (Complete Code)

import streamlit as st
import pandas as pd
from io import StringIO
# Import constants from the new utility file
from utils import REQUIRED_COLS 

st.set_page_config(layout="wide", page_title="Cricket Performance Dashboard")

st.title("Cricket Analysis Dashboard")

# Initialize session state for data sharing
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = pd.DataFrame()
    
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file:
    try:
        data = uploaded_file.getvalue().decode("utf-8")
        df_raw = pd.read_csv(StringIO(data))
        
        # Validation
        missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.session_state['df_raw'] = pd.DataFrame()
            st.stop()
            
        st.session_state['df_raw'] = df_raw
        st.sidebar.success("Data loaded successfully! Select a page to view the analysis.")

    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.session_state['df_raw'] = pd.DataFrame()
        
if st.session_state['df_raw'].empty:
    st.info("⬆️ Please upload a CSV file in the sidebar to begin the analysis.")
