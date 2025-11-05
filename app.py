import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# -------------------------------
# Page setup + custom styling
# -------------------------------
st.set_page_config(page_title="Cricket Analytics Dashboard", layout="wide")
st.markdown("""
<style>
.block-container {
    padding: 0.5rem 1rem 0.5rem 1rem;
    max-width: 1400px;
}
h1, h2, h3, h4 {
    text-align: center;
    font-weight: 700;
}
.batsman-header {
    font-size: 28px;
    font-weight: 700;
    color: #E28C1A; /* soft orange */
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# File upload
# -------------------------------
st.title("🏏 Cricket Analytics Dashboard")
uploaded_file = st.file_uploader("Upload your Hawkeye Data", type=["csv"])

if uploaded_file is None:
    st.info("👆 Please upload a CSV to start analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = [
    "BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", "BattingTeam",
    "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded",
    "BounceX", "BounceY", "LandingX", "LandingY",
    "InterceptionX", "InterceptionY", "InterceptionZ", "Over"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# -------------------------------
# Top filters
# -------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    team = st.selectbox("Batting Team", sorted(df["BattingTeam"].dropna().unique()))
    df = df[df["BattingTeam"] == team]
with col2:
    batsman = st.selectbox("Batsman", sorted(df["BatsmanName"].dropna().unique()))
    df = df[df["BatsmanName"] == batsman]
with col3:
    over = st.selectbox("Over", sorted(df["Over"].dropna().unique()))
    df = df[df["Over"] == over]

# fix delivery type to seam
df = df[df["DeliveryType"].str.lower().str.contains("seam")]

# batsman header
st.markdown(f"<div class='batsman-header'>{batsman}</div>", unsafe_allow_html=True)

# -------------------------------
# Chart 1: Crease Beehive Boxes
# -------------------------------
fig1, ax1 = plt.subplots(figsize=(6.5, 2.5))
for x in [-0.72, -0.45, -0.18, 0.18, 0.45, 0.72]:
    ax1.axvline(x, color="grey", lw=0.8, ls="--", alpha=0.6)
for y in [0, 0.71, 1.31, 1.91]:
    ax1.axhline(y, color="grey", lw=0.8, ls="--", alpha=0.6)
ax1.set_xlim(-0.75, 0.75)
ax1.set_ylim(0, 2)
ax1.axis("off")
st.pyplot(fig1, use_container_width=True)

# -------------------------------
# Chart 2: Crease Beehive
# -------------------------------
wickets = df[df["Wicket"] == True]
non_wickets = df[df["Wicket"] == False]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=non_wickets["StumpsY"], y=non_wickets["StumpsZ"],
    mode="markers",
    marker=dict(color="lightgrey", size=6, line=dict(color="white", width=0.4))
))
fig2.add_trace(go.Scatter(
    x=wickets["StumpsY"], y=wickets["StumpsZ"],
    mode="markers",
    marker=dict(color="red", size=9, line=dict(color="white", width=0.5))
))
for x in [-0.18, 0.18, -0.92, 0.92]:
    fig2.add_vline(x=x, line=dict(color="black", dash="dot", width=1))
fig2.update_layout(
    height=330, width=650,
    xaxis=dict(visible=False, range=[-1.2, 1.2]),
    yaxis=dict(visible=False, range=[0.4, 2]),
    margin=dict(l=10, r=10, t=20, b=10),
    plot_bgcolor="white", paper_bgcolor="white", showlegend=False
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Chart 3: Pitch Map
# -------------------------------
fig3 = go.Figure()
for y in [0.9, 2.8, 5.0, 8.6, 16.0]:
    fig3.add_hline(y=y, line=dict(color="grey", dash="dot", width=1))
p_w = df[df["Wicket"] == True]
p_nw = df[df["Wicket"] == False]
fig3.add_trace(go.Scatter(
    x=p_nw["BounceY"], y=p_nw["BounceX"], mode="markers",
    marker=dict(color="white", size=6, line=dict(color="grey", width=0.6))
))
fig3.add_trace(go.Scatter(
    x=p_w["BounceY"], y=p_w["BounceX"], mode="markers",
    marker=dict(color="red", size=9)
))
fig3.update_layout(
    height=360, width=650,
    xaxis=dict(visible=False, range=[-1.5, 1.5]),
    yaxis=dict(visible=False, range=[16, -4]),
    plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False
)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# Chart 4: Interception Side-on
# -------------------------------
df_int = df[df["InterceptionX"] > -999].copy()
df_int["Color"] = np.where(df_int["Wicket"], "red",
                           np.where(df_int["Runs"] >= 4, "royalblue", "lightgrey"))
fig4, ax4 = plt.subplots(figsize=(6.5, 2))
for c in ["lightgrey", "royalblue", "red"]:
    subset = df_int[df_int["Color"] == c]
    ax4.scatter(subset["InterceptionX"] + 10, subset["InterceptionZ"],
                s=25, color=c, edgecolors="black", linewidths=0.3)
for x in [0, 1.25, 2.0, 3.0]:
    ax4.axvline(x, color="grey", ls="--", lw=1)
ax4.set_xlim(-0.2, 3.4)
ax4.set_ylim(0, 1.5)
ax4.axis("off")
st.pyplot(fig4, use_container_width=True)

# -------------------------------
# Chart 5: Front-on + Wagon Wheel
# -------------------------------
colA, colB = st.columns([1, 1])

# --- Front-on Interception
with colA:
    fig5, ax5 = plt.subplots(figsize=(3, 2.8))
    for c in ["lightgrey", "royalblue", "red"]:
        subset = df_int[df_int["Color"] == c]
        ax5.scatter(subset["InterceptionY"], subset["InterceptionX"] + 10,
                    s=25, color=c, edgecolors="black", linewidths=0.3)
    for x in [-0.18, 0.18]:
        ax5.axvline(x, color="grey", lw=1)
    for y in [1.25, 2.0, 3.0]:
        ax5.axhline(y, color="grey", ls="--", lw=1)
    ax5.set_xlim(-1, 1)
    ax5.set_ylim(-0.2, 3.5)
    ax5.invert_yaxis()
    ax5.axis("off")
    st.pyplot(fig5, use_container_width=True)

# --- Scoring Areas (Wagon Wheel)
with colB:
    if "LandingX" in df.columns and "LandingY" in df.columns:
        df["Angle"] = np.degrees(np.arctan2(df["LandingY"], df["LandingX"]))
        bins = ["Fine Leg", "Square Leg", "Long On", "Long Off", "Cover", "Third Man"]
        df["Sector"] = pd.cut(df["Angle"], bins=len(bins), labels=bins)
        scoring = df.groupby("Sector")["Runs"].sum().reset_index()
        fig6, ax6 = plt.subplots(figsize=(3, 2.8))
        wedges, texts, autotexts = ax6.pie(
            scoring["Runs"], labels=scoring["Sector"], autopct='%1.0f%%',
            startangle=90, counterclock=False, wedgeprops={'width': 0.7}
        )
        for t in texts + autotexts:
            t.set_fontsize(8)
        ax6.axis("equal")
        st.pyplot(fig6, use_container_width=True)
    else:
        st.info("LandingX and LandingY missing for scoring wheel.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("All charts are proportionally aligned to resemble Tableau dashboard layout.")
