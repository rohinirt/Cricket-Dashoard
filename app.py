import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# ------------------------------
# PAGE CONFIG + CUSTOM STYLING
# ------------------------------
st.set_page_config(page_title="Cricket Analytics Dashboard", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        max-width: 1500px;
    }
    h1, h2, h3, h4, h5 {
        text-align: center;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# UPLOAD DATA
# ------------------------------
st.title("🏏 Cricket Analytics Dashboard")
uploaded_file = st.file_uploader("Upload your Hawkeye Data", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload your CSV to start analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = [
    "BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", 
    "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded",
    "BounceX", "BounceY", "LandingX", "LandingY", 
    "InterceptionX", "InterceptionY", "InterceptionZ", "Over"
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# ------------------------------
# TOP FILTER BAR
# ------------------------------
colf1, colf2, colf3 = st.columns([1, 1, 1])

with colf1:
    bat_team = st.selectbox("Select Batting Team", sorted(df["BattingTeam"].dropna().unique()))
    df = df[df["BattingTeam"] == bat_team]

with colf2:
    batsman = st.selectbox("Select Batsman", sorted(df["BatsmanName"].dropna().unique()))
    df = df[df["BatsmanName"] == batsman]

with colf3:
    over = st.selectbox("Select Over", sorted(df["Over"].dropna().unique()))
    df = df[df["Over"] == over]

# Fix delivery type to Seam
df = df[df["DeliveryType"].str.lower().str.contains("seam")]

# ------------------------------
# SECTION HEADING
# ------------------------------
st.markdown(f"## {batsman} — Performance Insights")

# ------------------------------
# CHART 1: CREASE BEEHIVE BOXES
# ------------------------------
st.markdown("### 1️⃣ Crease Beehive Boxes")

fig_box, ax_box = plt.subplots(figsize=(6, 3))

# Draw crease grid lines only (no color)
ax_box.plot([-0.72, 0.72], [0, 0], color="black", lw=1)
for x in [-0.72, -0.45, -0.18, 0.18, 0.45, 0.72]:
    ax_box.axvline(x, color="grey", lw=1, ls="--", alpha=0.5)
for y in [0, 0.71, 1.31, 1.91]:
    ax_box.axhline(y, color="grey", lw=1, ls="--", alpha=0.5)

ax_box.set_xlim(-0.75, 0.75)
ax_box.set_ylim(0, 2)
ax_box.axis("off")
st.pyplot(fig_box, use_container_width=True)

# ------------------------------
# CHART 2: CREASE BEEHIVE
# ------------------------------
st.markdown("### 2️⃣ Crease Beehive")

wickets = df[df["Wicket"] == True]
non_wickets = df[df["Wicket"] == False]

fig_cbh = go.Figure()
fig_cbh.add_trace(go.Scatter(
    x=non_wickets["StumpsY"], y=non_wickets["StumpsZ"],
    mode="markers",
    marker=dict(color="lightgrey", size=7, line=dict(color="white", width=0.5)),
    name="No Wicket"
))
fig_cbh.add_trace(go.Scatter(
    x=wickets["StumpsY"], y=wickets["StumpsZ"],
    mode="markers",
    marker=dict(color="red", size=10, line=dict(color="white", width=0.5)),
    name="Wicket"
))
# Add only crease lines (no background fill)
for x in [-0.18, 0.18, -0.92, 0.92]:
    fig_cbh.add_vline(x=x, line=dict(color="black", dash="dot", width=1))

fig_cbh.update_layout(
    height=400, width=600,
    xaxis=dict(visible=False, range=[-1.2, 1.2]),
    yaxis=dict(visible=False, range=[0.4, 2]),
    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    margin=dict(l=0, r=0, t=30, b=10)
)
st.plotly_chart(fig_cbh, use_container_width=True)

# ------------------------------
# CHART 3: PITCH MAP
# ------------------------------
st.markdown("### 3️⃣ Pitch Map")

fig_pitch = go.Figure()

# Horizontal separation lines
for y in [0.9, 2.8, 5.0, 8.6, 16.0]:
    fig_pitch.add_hline(y=y, line=dict(color="grey", dash="dot", width=1))

pitch_wickets = df[df["Wicket"] == True]
pitch_non_wickets = df[df["Wicket"] == False]

fig_pitch.add_trace(go.Scatter(
    x=pitch_non_wickets["BounceY"], y=pitch_non_wickets["BounceX"],
    mode="markers", name="No Wicket",
    marker=dict(color="white", size=7, line=dict(color="grey", width=0.8))
))
fig_pitch.add_trace(go.Scatter(
    x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"],
    mode="markers", name="Wicket",
    marker=dict(color="red", size=10)
))
fig_pitch.update_layout(
    height=450,
    xaxis=dict(visible=False, range=[-1.5, 1.5]),
    yaxis=dict(visible=False, range=[16, -4]),
    plot_bgcolor="white", margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False
)
st.plotly_chart(fig_pitch, use_container_width=True)

# ------------------------------
# CHART 4: INTERCEPTION SIDE-ON
# ------------------------------
st.markdown("### 4️⃣ Interception Points (Side-on View)")

df_inter = df[df["InterceptionX"] > -999].copy()
df_inter["Color"] = np.where(df_inter["Wicket"], "red", np.where(df_inter["Runs"] >= 4, "royalblue", "lightgrey"))

fig_side, ax_side = plt.subplots(figsize=(6, 2.5))
for c in ["lightgrey", "royalblue", "red"]:
    subset = df_inter[df_inter["Color"] == c]
    ax_side.scatter(subset["InterceptionX"] + 10, subset["InterceptionZ"], s=30, color=c, edgecolors="black", linewidths=0.3)

# Draw lines
for x in [0, 1.25, 2.0, 3.0]:
    ax_side.axvline(x, color="grey", linestyle="--", lw=1)
ax_side.set_xlim(-0.2, 3.4)
ax_side.set_ylim(0, 1.5)
ax_side.axis("off")
st.pyplot(fig_side, use_container_width=True)

# ------------------------------
# CHART 5: INTERCEPTION FRONT-ON + SCORING AREAS
# ------------------------------
st.markdown("### 5️⃣ Interception (Front-on) and Scoring Areas")

colA, colB = st.columns([1, 1])

# Front-on view
with colA:
    fig_front, ax_front = plt.subplots(figsize=(3, 3))
    for c in ["lightgrey", "royalblue", "red"]:
        subset = df_inter[df_inter["Color"] == c]
        ax_front.scatter(subset["InterceptionY"], subset["InterceptionX"] + 10, s=30, color=c, edgecolors="black", linewidths=0.3)
    for x in [-0.18, 0.18]:
        ax_front.axvline(x, color="grey", lw=1)
    for y in [1.25, 2.0, 3.0]:
        ax_front.axhline(y, color="grey", linestyle="--", lw=1)
    ax_front.set_xlim(-1, 1)
    ax_front.set_ylim(-0.2, 3.5)
    ax_front.invert_yaxis()
    ax_front.axis("off")
    st.pyplot(fig_front, use_container_width=True)

# Scoring Areas Pie
with colB:
    st.markdown("#### Scoring Areas (Wagon Wheel %)")
    if "LandingX" in df.columns and "LandingY" in df.columns:
        df["Angle"] = np.degrees(np.arctan2(df["LandingY"], df["LandingX"]))
        bins = ["Fine Leg", "Square Leg", "Long On", "Long Off", "Cover", "Third Man"]
        df["Sector"] = pd.cut(df["Angle"], bins=len(bins), labels=bins)
        scoring = df.groupby("Sector")["Runs"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(scoring["Runs"], labels=scoring["Sector"], autopct='%1.0f%%', startangle=90, counterclock=False)
        st.pyplot(fig)
    else:
        st.info("LandingX and LandingY missing for scoring wheel.")

st.markdown("---")
st.caption("Designed for compact layout — All charts aligned vertically and consistently sized.")
