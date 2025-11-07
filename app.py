import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from io import StringIO

# -------------------------------------------------------------------------
# PAGE SELECTOR - this should be placed right after your imports
# -------------------------------------------------------------------------
page = st.sidebar.radio("Select page", ["Batters (current)", "Pacers", "Spinners"], index=0)
st.sidebar.markdown("Select a page to view Pacers / Spinners analysis")

# -------------------------------------------------------------------------
# SHARED FUNCTIONS AND GLOBAL SETUP
# (you already have these in your file: create_crease_beehive, create_pitch_map, etc.)
# Keep them outside the page if-blocks so all pages can use them.
# -------------------------------------------------------------------------
REQUIRED_COLS = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ",
                 "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded",
                 "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX",
                 "InterceptionY", "InterceptionZ", "Over"]

# Example small helper
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# -------------------------------------------------------------------------
# PAGE 1: BATTERS (your existing main dashboard)
# -------------------------------------------------------------------------
if page == "Batters (current)":
    st.title("Batters Dashboard")
    uploaded_file = st.file_uploader("Upload Hawkeye CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis.")
        st.stop()

    df_raw = load_csv(uploaded_file)
    if df_raw is None or not all(col in df_raw.columns for col in REQUIRED_COLS):
        st.error("Missing required columns.")
        st.stop()

    # 🔹 reuse your current filters
    all_teams = ["All"] + sorted(df_raw["BattingTeam"].dropna().unique().tolist())
    team = st.selectbox("Batting Team", all_teams)
    batsmen = ["All"] + sorted(df_raw["BatsmanName"].dropna().unique().tolist())
    batsman = st.selectbox("Batsman", batsmen)

    # 🔹 Apply filters
    df_seam = df_raw[df_raw["DeliveryType"] == "Seam"]
    df_spin = df_raw[df_raw["DeliveryType"] == "Spin"]

    st.header(batsman if batsman != "All" else "Global Batting Analysis")

    # 🔹 Your existing code for charts goes here, e.g.:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_crease_beehive(df_seam, "Seam"), use_container_width=True)
        st.plotly_chart(create_pitch_map(df_seam, "Seam"), use_container_width=True)
    with col2:
        st.plotly_chart(create_crease_beehive(df_spin, "Spin"), use_container_width=True)
        st.plotly_chart(create_pitch_map(df_spin, "Spin"), use_container_width=True)

# -------------------------------------------------------------------------
# PAGE 2: PACERS
# -------------------------------------------------------------------------
elif page == "Pacers":
    st.title("Pacers Dashboard")
    uploaded_file = st.file_uploader("Upload CSV for Pacers", type=["csv"], key="pacer_file")
    if uploaded_file:
        df_raw = load_csv(uploaded_file)
        df_pacers = df_raw[df_raw["DeliveryType"].str.lower().str.contains("seam|pacer", na=False)]
        st.plotly_chart(create_crease_beehive(df_pacers, "Seam"), use_container_width=True)
        st.plotly_chart(create_pitch_map(df_pacers, "Seam"), use_container_width=True)
    else:
        st.info("Upload CSV to view Pacers dashboard.")

# -------------------------------------------------------------------------
# PAGE 3: SPINNERS
# -------------------------------------------------------------------------
elif page == "Spinners":
    st.title("Spinners Dashboard")
    uploaded_file = st.file_uploader("Upload CSV for Spinners", type=["csv"], key="spin_file")
    if uploaded_file:
        df_raw = load_csv(uploaded_file)
        df_spinners = df_raw[df_raw["DeliveryType"].str.lower().str.contains("spin", na=False)]
        st.plotly_chart(create_crease_beehive(df_spinners, "Spin"), use_container_width=True)
        st.plotly_chart(create_pitch_map(df_spinners, "Spin"), use_container_width=True)
    else:
        st.info("Upload CSV to view Spinners dashboard.")
