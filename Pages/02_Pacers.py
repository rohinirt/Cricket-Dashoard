# pages/02_Pacers.py - (Complete Code)

import streamlit as st
import pandas as pd
# Import the chart function you want to test first
from utils import create_crease_beehive 

st.title("⚾ Pacers (Seam) Analysis")

df_raw = st.session_state.get('df_raw')

if df_raw.empty:
    st.warning("Please upload a file via the main page to continue.")
    st.stop()

# 1. Filter for Pacers (Seam)
df_pacer = df_raw[df_raw["DeliveryType"] == "Seam"].copy()

# 2. Select a Bowler
bowlers = ["All"] + sorted(df_pacer["BowlerName"].dropna().unique().tolist())
selected_bowler = st.selectbox("Select Pacer", bowlers)

# 3. Apply Filter
if selected_bowler != "All":
    df_pacer = df_pacer[df_pacer["BowlerName"] == selected_bowler]

st.markdown("---")
st.header(f"Pitch Map Test for: {selected_bowler.upper()}")

# 4. Trial & Error: Start with one working chart
# NOTE: This uses the existing Batters chart, which may need modifications 
# to display bowler data correctly, but it's a good starting point.
if not df_pacer.empty:
    st.plotly_chart(create_crease_beehive(df_pacer, "Seam"), use_container_width=True)
else:
    st.info("No Pacer data available for selected filter.")
