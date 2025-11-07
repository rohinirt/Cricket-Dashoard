import streamlit as st
import pandas as pd
import numpy as np

# Import ALL necessary functions and constants from the utility file
from utils import (
    REQUIRED_COLS, 
    fig_to_png_bytes, 
    create_pdf_report, 
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

# --- PAGE SETUP & DATA RETRIEVAL ---
st.title("🏏 Batters Performance Analysis")

# Retrieve the raw data frame shared from app.py via session_state
df_raw = st.session_state.get('df_raw')

if df_raw.empty:
    st.warning("Please upload a file via the main page to continue.")
    st.stop()

# Data separation
df_seam_raw = df_raw[df_raw["DeliveryType"] == "Seam"].copy()
df_spin_raw = df_raw[df_raw["DeliveryType"] == "Spin"].copy()

# --- FILTERS ---
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

innings_options = ["All"] + sorted(df_raw["Innings"].dropna().unique().tolist())
with filter_col3:
    selected_innings = st.selectbox("Innings", innings_options, index=0)

bowler_hand_options = ["All", "Right Hand", "Left Hand"]
with filter_col4:
    selected_bowler_hand = st.selectbox("Bowler Hand", bowler_hand_options, index=0)

# --- Apply Filters Function ---
def apply_filters(df):
    if bat_team != "All":
        df = df[df["BattingTeam"] == bat_team]
    if batsman != "All":
        df = df[df["BatsmanName"] == batsman]
    if selected_innings != "All":
        df = df[df["Innings"] == selected_innings]
    if selected_bowler_hand != "All":
        is_right = (selected_bowler_hand == "Right Hand")
        df = df[df["IsBowlerRightHanded"] == is_right]
    return df

df_seam = apply_filters(df_seam_raw)
df_spin = apply_filters(df_spin_raw)

# --- CHART GENERATION (Create all figures) ---
figures = {
    'seam_zonal': create_zonal_analysis(df_seam, batsman, "Seam"),
    'seam_beehive': create_crease_beehive(df_seam, "Seam"),
    'seam_lateral': create_lateral_performance_boxes(df_seam, "Seam", batsman),
    'seam_pitchmap': create_pitch_map(df_seam, "Seam"),
    'seam_pitch_pct': create_pitch_length_run_pct(df_seam, "Seam"),
    'seam_side_on': create_interception_side_on(df_seam, "Seam"),
    'seam_crease_split': create_crease_width_split(df_seam, "Seam"),
    'seam_top_on': create_interception_front_on(df_seam, "Seam"),
    'seam_wagon_wheel': create_wagon_wheel(df_seam, "Seam"),
    'seam_lr_split': create_left_right_split(df_seam, "Seam"),
    'seam_swing': create_directional_split(df_seam, "Swing", "Swing", "Seam"),
    'seam_deviation': create_directional_split(df_seam, "Deviation", "Deviation", "Seam"),
    
    'spin_zonal': create_zonal_analysis(df_spin, batsman, "Spin"),
    'spin_beehive': create_crease_beehive(df_spin, "Spin"),
    'spin_lateral': create_lateral_performance_boxes(df_spin, "Spin", batsman),
    'spin_pitchmap': create_pitch_map(df_spin, "Spin"),
    'spin_pitch_pct': create_pitch_length_run_pct(df_spin, "Spin"),
    'spin_side_on': create_interception_side_on(df_spin, "Spin"),
    'spin_crease_split': create_crease_width_split(df_spin, "Spin"),
    'spin_top_on': create_interception_front_on(df_spin, "Spin"),
    'spin_wagon_wheel': create_wagon_wheel(df_spin, "Spin"),
    'spin_lr_split': create_left_right_split(df_spin, "Spin"),
    'spin_swing': create_directional_split(df_spin, "Swing", "Drift", "Spin"),
    'spin_deviation': create_directional_split(df_spin, "Deviation", "Turn", "Spin")
}

# --- DOWNLOAD BUTTON & DISPLAY ---
heading_col, download_col = st.columns([4, 1])

heading_text = batsman.upper() if batsman != "All" else "GLOBAL ANALYSIS"
with heading_col:
    st.header(f"**{heading_text}**")

with download_col:
    # Use the imported function from utils.py
    pdf_bytes = create_pdf_report(figures, "BATTER", heading_text) 
    
    st.download_button(
        label="Download PDF Report ⬇️",
        data=pdf_bytes,
        file_name=f"Report_Batter_{heading_text.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

# --- DISPLAY CHARTS ---
col1, col2 = st.columns(2)

# --- LEFT COLUMN: SEAM ANALYSIS ---
with col1:
    st.markdown("#### SEAM")

    st.markdown("###### CREASE BEEHIVE ZONES")
    st.pyplot(figures['seam_zonal'], use_container_width=True)
    
    st.markdown("###### CREASE BEEHIVE")
    st.plotly_chart(figures['seam_beehive'], use_container_width=True)

    st.pyplot(figures['seam_lateral'], use_container_width=True)
    
    pitch_map_col, run_pct_col = st.columns([3, 1])

    with pitch_map_col:
        st.markdown("###### PITCHMAP")
        st.plotly_chart(figures['seam_pitchmap'], use_container_width=True)
        
    with run_pct_col:
        st.markdown("###### ")
        st.pyplot(figures['seam_pitch_pct'], use_container_width=True)
    
    st.markdown("###### INTERCEPTION SIDE-ON")
    st.pyplot(figures['seam_side_on'], use_container_width=True)

    st.pyplot(figures['seam_crease_split'], use_container_width=True)

    bottom_col_left, bottom_col_right = st.columns(2)

    with bottom_col_left:
        st.markdown("###### INTERCEPTION TOP-ON")
        st.pyplot(figures['seam_top_on'], use_container_width=True)
    
    with bottom_col_right:
        st.markdown("###### SCORING AREAS")    
        st.pyplot(figures['seam_wagon_wheel'], use_container_width=True)
        st.pyplot(figures['seam_lr_split'], use_container_width=True)

    final_col_swing, final_col_deviation = st.columns(2)

    with final_col_swing:
        st.pyplot(figures['seam_swing'], use_container_width=True)

    with final_col_deviation:
        st.pyplot(figures['seam_deviation'], use_container_width=True)   

# --- RIGHT COLUMN: SPIN ANALYSIS ---
with col2:
    st.markdown("#### SPIN")
    
    st.markdown("###### CREASE BEEHIVE ZONES")
    st.pyplot(figures['spin_zonal'], use_container_width=True)
    st.markdown("###### CREASE BEEHIVE")
    st.plotly_chart(figures['spin_beehive'], use_container_width=True)

    st.pyplot(figures['spin_lateral'], use_container_width=True)

    pitch_map_col, run_pct_col = st.columns([3, 1])

    with pitch_map_col:
        st.markdown("###### PITCHMAP")
        st.plotly_chart(figures['spin_pitchmap'], use_container_width=True) 
        
    with run_pct_col:
        st.markdown("###### ")
        st.pyplot(figures['spin_pitch_pct'], use_container_width=True)
    
    st.markdown("###### INTERCEPTION SIDE-ON")
    st.pyplot(figures['spin_side_on'], use_container_width=True)

    st.pyplot(figures['spin_crease_split'], use_container_width=True)

    bottom_col_left, bottom_col_right = st.columns(2)

    with bottom_col_left:
        st.markdown("###### INTERCEPTION TOP-ON")
        st.pyplot(figures['spin_top_on'], use_container_width=True)
    
    with bottom_col_right:
        st.markdown("###### SCORING AREAS")
        st.pyplot(figures['spin_wagon_wheel'], use_container_width=True)
        st.pyplot(figures['spin_lr_split'], use_container_width=True)
        
    final_col_swing, final_col_deviation = st.columns(2)

    with final_col_swing:
        st.pyplot(figures['spin_swing'], use_container_width=True)

    with final_col_deviation:
        st.pyplot(figures['spin_deviation'], use_container_width=True)
