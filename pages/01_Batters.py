import streamlit as st
import pandas as pd
import sys

# Import all utility functions
try:
    from utils import (
        create_zonal_analysis, 
        create_lateral_performance_boxes, 
        create_crease_beehive, 
        create_pitch_map, 
        create_pitch_length_run_pct, 
        create_interception_side_on, 
        create_crease_width_split, 
        create_interception_front_on, 
        create_wagon_wheel, 
        create_left_right_split, 
        create_directional_split
    )
except ImportError:
    st.error("Error: Could not import utility functions from utils.py. Check your file structure.")
    sys.exit()

# --- PAGE SETUP ---
st.title("🏏 Trial Batters Dashboard")

# 1. Retrieve the data shared from the main app.py file
if 'df_raw' not in st.session_state or st.session_state['df_raw'].empty:
    st.warning("Please upload a CSV file via the main page to continue.")
    st.stop() 

df_raw = st.session_state['df_raw']

# 2. Data Separation (Seam/Spin) - Simplified for trial
# Assuming "Seam" and "Spin" exist, otherwise one dataframe will be empty.
df_seam_raw = df_raw[df_raw["DeliveryType"] == "Seam"].copy()
df_spin_raw = df_raw[df_raw["DeliveryType"] == "Spin"].copy()


# 3. --- FILTERS START HERE ---
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4) 

all_teams = ["All"] + sorted(df_raw["BattingTeam"].dropna().unique().tolist())
with filter_col1:
    bat_team = st.selectbox("Batting Team", all_teams, index=0)

if bat_team != "All":
    batsmen_options = ["All"] + sorted(df_raw[df_raw["BattingTeam"] == bat_team]["BatsmanName"].dropna().unique().tolist())
else:
    batsmen_options = ["All"] + sorted(df_raw["BatsmanName"].dropna().unique().tolist())

with filter_col2:
    batsman = st.selectbox("Batsman Name", batsmen_options, index=0)

innings_options = ["All"] + sorted(df_raw["Innings"].dropna().astype(int).astype(str).unique().tolist())
with filter_col3:
    selected_innings = st.selectbox("Innings", innings_options, index=0)

bowler_hand_options = ["All", "Right Hand", "Left Hand"]
with filter_col4:
    selected_bowler_hand = st.selectbox("Bowler Hand", bowler_hand_options, index=0)


# 4. --- Apply Filters Function ---
def apply_filters(df):
    df_filtered = df.copy()
    if bat_team != "All":
        df_filtered = df_filtered[df_filtered["BattingTeam"] == bat_team]
    if batsman != "All":
        df_filtered = df_filtered[df_filtered["BatsmanName"] == batsman]
    
    if selected_innings != "All":
        try:
            target_inning = int(selected_innings)
            df_filtered = df_filtered[df_filtered["Innings"] == target_inning]
        except ValueError:
            pass 

    if selected_bowler_hand != "All":
        is_right = (selected_bowler_hand == "Right Hand")
        # Note: IsBowlerRightHanded may not exist in the trial data, use a try/except or a check if it causes errors
        # if 'IsBowlerRightHanded' in df_filtered.columns:
        #     df_filtered = df_filtered[df_filtered["IsBowlerRightHanded"] == is_right]
        pass # Skip this filter for the simplified trial
    return df_filtered

# 5. Apply Filters to DataFrames
df_seam = apply_filters(df_seam_raw)
df_spin = apply_filters(df_spin_raw)

# 6. --- CHART GENERATION ---
figures = {
    # Simple Bar Chart (Plotly)
    'seam_runs': create_zonal_analysis(df_seam, batsman, "Seam"), 
    'spin_runs': create_zonal_analysis(df_spin, batsman, "Spin"),
    
    # Simple Pie Chart (Matplotlib)
    'seam_balls': create_directional_split(df_seam, "Swing", "SWING (Balls)", "Seam"),
    'spin_balls': create_directional_split(df_spin, "Deviation", "TURN (Balls)", "Spin"), 
    
    # Placeholders (Matplotlib/Plotly)
    'seam_lateral': create_lateral_performance_boxes(df_seam, "Seam", batsman),
    'seam_beehive': create_crease_beehive(df_seam, "Seam"),
    'seam_pitchmap': create_pitch_map(df_seam, "Seam"),
    'spin_beehive': create_crease_beehive(df_spin, "Spin"),
    'spin_pitchmap': create_pitch_map(df_spin, "Spin"),
}

# 7. --- HEADING DISPLAY ---
st.header(f"Analysis for: {batsman.upper()}")
st.markdown("---")

# --- DISPLAY CHARTS ---
col1, col2 = st.columns(2)

# --- LEFT COLUMN: SEAM ANALYSIS ---
with col1:
    st.markdown("#### SEAM ANALYSIS")

    st.markdown("###### TRIAL CHART 1: TOTAL RUNS (Plotly)")
    st.plotly_chart(figures['seam_runs'], use_container_width=True)
    
    st.markdown("###### TRIAL CHART 2: BALLS BY INNINGS (Matplotlib)")
    st.pyplot(figures['seam_balls'], use_container_width=True)
    
    st.markdown("###### PLACEHOLDER: LATERAL BOXES (Matplotlib)")
    st.pyplot(figures['seam_lateral'], use_container_width=True)
    
    st.markdown("###### PLACEHOLDER: BEEHIVE (Plotly)")
    st.plotly_chart(figures['seam_beehive'], use_container_width=True)


# --- RIGHT COLUMN: SPIN ANALYSIS ---
with col2:
    st.markdown("#### SPIN ANALYSIS")

    st.markdown("###### TRIAL CHART 1: TOTAL RUNS (Plotly)")
    st.plotly_chart(figures['spin_runs'], use_container_width=True)

    st.markdown("###### TRIAL CHART 2: BALLS BY INNINGS (Matplotlib)")
    st.pyplot(figures['spin_balls'], use_container_width=True)
    
    st.markdown("###### PLACEHOLDER: PITCHMAP (Plotly)")
    st.plotly_chart(figures['spin_pitchmap'], use_container_width=True)
    
    st.markdown("###### PLACEHOLDER: BEEHIVE (Plotly)")
    st.plotly_chart(figures['spin_beehive'], use_container_width=True)
