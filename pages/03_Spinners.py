# pages/03_Spinners.py - (Complete Code)

import streamlit as st
import pandas as pd
# Import the chart function you want to test first
from utils import create_zonal_analysis

st.title("🌀 Spinners Analysis")

df_raw = st.session_state.get('df_raw')

if df_raw.empty:
    st.warning("Please upload a file via the main page to continue.")
    st.stop()

# 1. Filter for Spinners
df_spinner = df_raw[df_raw["DeliveryType"] == "Spin"].copy()

# 2. Select a Bowler
bowlers = ["All"] + sorted(df_spinner["BowlerName"].dropna().unique().tolist())
selected_bowler = st.selectbox("Select Spinner", bowlers)

# 3. Apply Filter
if selected_bowler != "All":
    df_spinner = df_spinner[df_spinner["BowlerName"] == selected_bowler]

st.markdown("---")
st.header(f"Zonal Map Test for: {selected_bowler.upper()}")

# 4. Trial & Error: Start with one working chart
# NOTE: Using the Batters chart as a starting point.
if not df_spinner.empty:
    st.pyplot(create_zonal_analysis(df_spinner, selected_bowler, "Spin"), use_container_width=True)
else:
    st.info("No Spinner data available for selected filter.")
