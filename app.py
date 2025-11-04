import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- 1. SET PAGE CONFIGURATION ---

# --- 2. DATA UPLOADER AND INITIAL LOAD ---
# Use the main area for the file uploader for better visibility
st.title("Cricket Analysis Dashboard 🏏")
st.markdown("Upload your Hawkeye data (CSV) below to begin analysis.")
uploaded_file = st.file_uploader("Upload your data", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # --- Logic removed here ---
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None # Ensure df is None if reading fails

if df is None:
    st.info("👆 Please upload a CSV file with the required Hawkeye data columns to view the dashboard.")
    st.stop() # Stop execution if data is not loaded

# --- 3. SIDEBAR FILTERS (COMMON TO BOTH CHARTS) ---
st.sidebar.header("Data Filters 📊")

# Ensure all filter columns are available before creating options
if "BattingTeam" not in df.columns:
    st.error("Column 'BattingTeam' not found for filtering.")
    st.stop()

# Filter 1: Batting Team
bat_team_options = ["All"] + sorted(df["BattingTeam"].dropna().unique().tolist())
bat_team = st.sidebar.selectbox("Select Batting Team", bat_team_options)
df_bat_team = df if bat_team == "All" else df[df["BattingTeam"] == bat_team]

# Filter 2: Batsman Name (Cascading based on Batting Team)
batsman_options = ["All"] + sorted(df_bat_team["BatsmanName"].dropna().unique().tolist())
batsman = st.sidebar.selectbox("Select Batsman", batsman_options)
df_batsman = df_bat_team if batsman == "All" else df_bat_team[df_bat_team["BatsmanName"] == batsman]

# Filter 3: Delivery Type (Cascading based on Batsman)
delivery_options = ["All"] + sorted(df_batsman["DeliveryType"].dropna().unique().tolist())
delivery = st.sidebar.selectbox("Select Delivery Type", delivery_options)
filtered_df = df_batsman if delivery == "All" else df_batsman[df_batsman["DeliveryType"] == delivery]


# --- Check for required columns ---
required_cols = ["LandingX", "LandingY", "IsBatsmanRightHanded", "Runs"]
if not all(col in df.columns for col in required_cols):
    st.warning(f"Missing columns for wagon wheel. Required: {required_cols}")
else:
    # --- Apply Tableau logic ---
    def scoring_wagon(row):
        x, y = row["LandingX"], row["LandingY"]
        right = row["IsBatsmanRightHanded"]
        angle = np.arctan2(y, x)

        if right:
            if x <= 0 and y > 0:
                return "FINE LEG"
            elif x <= 0 and y <= 0:
                return "THIRD MAN"
            elif x > 0 and y < 0 and angle < np.pi / -4:
                return "COVER"
            elif x > 0 and y < 0 and np.arctan(x / y) <= np.pi / -4:
                return "LONG OFF"
            elif x > 0 and y >= 0 and angle >= np.pi / 4:
                return "SQUARE LEG"
            elif x > 0 and y >= 0 and angle <= np.pi / 4:
                return "LONG ON"
        else:
            if x <= 0 and y > 0:
                return "THIRD MAN"
            elif x <= 0 and y <= 0:
                return "FINE LEG"
            elif x > 0 and y < 0 and angle < np.pi / -4:
                return "SQUARE LEG"
            elif x > 0 and y < 0 and np.arctan(x / y) <= np.pi / -4:
                return "LONG ON"
            elif x > 0 and y >= 0 and angle >= np.pi / 4:
                return "COVER"
            elif x > 0 and y >= 0 and angle <= np.pi / 4:
                return "LONG OFF"
        return None

    df["ScoringWagon"] = df.apply(scoring_wagon, axis=1)

    # --- Compute total runs and % ---
    agg_df = df.groupby("ScoringWagon", dropna=True)["Runs"].sum().reset_index()
    agg_df["Percentage"] = (agg_df["Runs"] / agg_df["Runs"].sum()) * 100

    # --- Assign fixed angles to zones ---
    angle_map = {
        "FINE LEG": 135,
        "THIRD MAN": -135,
        "COVER": -45,
        "SQUARE LEG": 45,
        "LONG OFF": -15,
        "LONG ON": 15
    }
    agg_df["Angle"] = agg_df["ScoringWagon"].map(angle_map)

    # --- Wagon Wheel Plot ---
    wagon = go.Figure()
    colors = ['#e63946', '#f3722c', '#f9c74f', '#90be6d', '#43aa8b', '#577590']

    for i, r in agg_df.iterrows():
        wagon.add_trace(go.Scatterpolar(
            r=[0, r["Runs"]],
            theta=[0, r["Angle"]],
            mode="lines+markers+text",
            line=dict(width=4, color=colors[i % len(colors)]),
            text=[None, f"{r['ScoringWagon']}<br>{r['Percentage']:.1f}%"],
            textposition="top center",
            marker=dict(size=8, color=colors[i % len(colors)], line=dict(color="white", width=1.5))
        ))

    wagon.update_layout(
        title=f"{BatsmanName} - Wagon Wheel",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, agg_df["Runs"].max() * 1.2]),
            angularaxis=dict(direction="clockwise", rotation=90)
        ),
        showlegend=False,
        height=500,
        width=500
    )

    # --- Pie Chart ---
    pie = go.Figure(data=[
        go.Pie(
            labels=agg_df["ScoringWagon"],
            values=agg_df["Runs"],
            hole=0.3,
            textinfo="label+percent",
            marker=dict(colors=colors, line=dict(color="white", width=1))
        )
    ])
    pie.update_layout(title="Scoring Areas Distribution", height=500, width=500)

    # --- Display Both ---
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(wagon, use_container_width=True)
    with col2:
        st.plotly_chart(pie, use_container_width=True)


