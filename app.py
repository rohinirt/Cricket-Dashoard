import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_elements import elements, dashboard, mui, echart

# =====================================
# 1️⃣ PAGE CONFIGURATION
# =====================================
st.set_page_config(page_title="Cricket Analytics Dashboard", layout="wide")
st.title("🏏 Cricket Analytics Dashboard (Powered by streamlit-elements)")

# =====================================
# 2️⃣ DATA UPLOAD
# =====================================
uploaded_file = st.file_uploader("Upload your Hawkeye data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# =====================================
# 3️⃣ FILTERS
# =====================================
st.sidebar.header("🎯 Filters")

bat_team_options = ["All"] + sorted(df["BattingTeam"].dropna().unique().tolist())
bat_team = st.sidebar.selectbox("Select Batting Team", bat_team_options)
df_bat_team = df if bat_team == "All" else df[df["BattingTeam"] == bat_team]

batsman_options = ["All"] + sorted(df_bat_team["BatsmanName"].dropna().unique().tolist())
batsman = st.sidebar.selectbox("Select Batsman", batsman_options)
df_batsman = df_bat_team if batsman == "All" else df_bat_team[df_bat_team["BatsmanName"] == batsman]

delivery_options = ["All"] + sorted(df_batsman["DeliveryType"].dropna().unique().tolist())
delivery = st.sidebar.selectbox("Select Delivery Type", delivery_options)
filtered_df = df_batsman if delivery == "All" else df_batsman[df_batsman["DeliveryType"] == delivery]

if filtered_df.empty:
    st.error("No data matches the selected filters.")
    st.stop()

# =====================================
# 4️⃣ PREPARE SOME SAMPLE CHART DATA
# =====================================
# Example fallback for demonstration
zone_data = {"Zone": ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], "Runs": np.random.randint(5, 40, 6)}
pitch_data = {"Length": ["Short", "Length", "Slot", "Yorker", "Full Toss"], "Count": [12, 20, 9, 15, 8]}
wagon_data = {"Angle": [30, 60, 120, 200, 300, 340], "Runs": [20, 35, 40, 25, 10, 30]}

zone_df = pd.DataFrame(zone_data)
pitch_df = pd.DataFrame(pitch_data)
wagon_df = pd.DataFrame(wagon_data)

# =====================================
# 5️⃣ STREAMLIT-ELEMENTS DASHBOARD GRID
# =====================================

layout = [
    dashboard.Item("cbh", 0, 0, 6, 3),     # Chart 1
    dashboard.Item("zone", 6, 0, 6, 3),    # Chart 2
    dashboard.Item("pitch", 0, 3, 6, 3),   # Chart 3
    dashboard.Item("wagon", 6, 3, 6, 3),   # Chart 4
]

with elements("dashboard"):
    with dashboard.Grid(layout, draggable=True, resizable=True):

        # ========== Chart 1: Crease Beehive ==========
        with mui.Card(key="cbh", sx={"p": 2, "m": 1, "height": "100%"}):
            mui.Typography("Crease Beehive", variant="h6", sx={"mb": 1})
            echart(
                options={
                    "xAxis": {"show": False},
                    "yAxis": {"show": False},
                    "series": [{
                        "symbolSize": 8,
                        "data": [[-0.2, 1.5], [0.4, 1.8], [-0.5, 1.2]],
                        "type": "scatter",
                        "itemStyle": {"color": "#e63946"},
                    }],
                },
                height="250px",
            )

        # ========== Chart 2: Zone Heatmap ==========
        with mui.Card(key="zone", sx={"p": 2, "m": 1, "height": "100%"}):
            mui.Typography("Zonal Run Distribution", variant="h6", sx={"mb": 1})
            echart(
                options={
                    "xAxis": {"type": "category", "data": zone_df["Zone"].tolist()},
                    "yAxis": {"type": "value"},
                    "series": [{"data": zone_df["Runs"].tolist(), "type": "bar", "color": "#457b9d"}],
                },
                height="250px",
            )

        # ========== Chart 3: Pitch Map ==========
        with mui.Card(key="pitch", sx={"p": 2, "m": 1, "height": "100%"}):
            mui.Typography("Pitch Map", variant="h6", sx={"mb": 1})
            echart(
                options={
                    "xAxis": {"type": "category", "data": pitch_df["Length"].tolist()},
                    "yAxis": {"type": "value"},
                    "series": [{"data": pitch_df["Count"].tolist(), "type": "bar", "color": "#f4a261"}],
                },
                height="250px",
            )

        # ========== Chart 4: Wagon Wheel ==========
        with mui.Card(key="wagon", sx={"p": 2, "m": 1, "height": "100%"}):
            mui.Typography("Wagon Wheel", variant="h6", sx={"mb": 1})
            echart(
                options={
                    "polar": {"radius": [20, "80%"]},
                    "angleAxis": {"type": "value", "startAngle": 90},
                    "radiusAxis": {"show": False},
                    "series": [{
                        "type": "bar",
                        "data": wagon_df["Runs"].tolist(),
                        "coordinateSystem": "polar",
                        "name": "Runs",
                        "color": "#2a9d8f"
                    }],
                },
                height="250px",
            )
