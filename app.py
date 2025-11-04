import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.subheader("🏏 Wagon Wheel and Scoring Areas")
uploaded_file = st.file_uploader("Upload your data", type=["csv"])
# --- Sample data or your uploaded data ---
df = pd.read_csv(uploaded_file)
# Expected columns: LandingX, LandingY, IsBatsmanRightHanded, Runs
# For testing, generate small dummy data

# --- Apply Tableau logic in Python ---
def scoring_wagon(row):
    x, y = row['LandingX'], row['LandingY']
    right = row['IsBatsmanRightHanded']
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

# --- Compute total runs by area ---
agg_df = df.groupby("ScoringWagon", dropna=True)["Runs"].sum().reset_index()
agg_df["Percentage"] = (agg_df["Runs"] / agg_df["Runs"].sum()) * 100

# --- Assign angles to areas (for visual placement) ---
angle_map = {
    "FINE LEG": 135,
    "THIRD MAN": -135,
    "COVER": -45,
    "SQUARE LEG": 45,
    "LONG OFF": -15,
    "LONG ON": 15
}
agg_df["Angle"] = agg_df["ScoringWagon"].map(angle_map)

# --- Wagon wheel plot ---
wagon = go.Figure()

for _, r in agg_df.iterrows():
    wagon.add_trace(go.Scatterpolar(
        r=[0, r["Runs"]],
        theta=[0, r["Angle"]],
        mode='lines+markers+text',
        line=dict(width=3, color='crimson'),
        text=[None, f"{r['ScoringWagon']}<br>{r['Percentage']:.1f}%"],
        textposition='top center'
    ))

wagon.update_layout(
    polar=dict(
        radialaxis=dict(visible=False, range=[0, agg_df["Runs"].max() + 10]),
        angularaxis=dict(direction="clockwise", rotation=90)
    ),
    showlegend=False,
    title="Batting Wagon Wheel"
)

# --- Pie chart for scoring area distribution ---
pie = go.Figure(
    data=[
        go.Pie(
            labels=agg_df["ScoringWagon"],
            values=agg_df["Runs"],
            hole=0.3,
            textinfo='label+percent',
            marker=dict(colors=['#f94144', '#f3722c', '#f9c74f', '#90be6d', '#43aa8b', '#577590'])
        )
    ]
)
pie.update_layout(title="Scoring Areas Distribution")

# --- Layout display ---
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(wagon, use_container_width=True)
with col2:
    st.plotly_chart(pie, use_container_width=True)
