import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

st.set_page_config(page_title="Cricket Analytics Dashboard", layout="wide")

# --------------------------
# Load Data (Upload or Default)
# --------------------------
st.sidebar.header("Upload or Use Default Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("/mnt/data/queryOutput_1751703447858.csv")

# --------------------------
# Common Filters (Batting Team → Batsman → Delivery Type)
# --------------------------
st.sidebar.header("Filters")

bat_team_options = ["All"] + sorted(df["BattingTeam"].dropna().unique())
bat_team = st.sidebar.selectbox("Select Batting Team", bat_team_options)
df_bat_team = df if bat_team == "All" else df[df["BattingTeam"] == bat_team]

batsman_options = ["All"] + sorted(df_bat_team["BatsmanName"].dropna().unique())
batsman = st.sidebar.selectbox("Select Batsman", batsman_options)
df_batsman = df_bat_team if batsman == "All" else df_bat_team[df_bat_team["BatsmanName"] == batsman]

delivery_options = ["All"] + sorted(df_batsman["DeliveryType"].dropna().unique())
delivery = st.sidebar.selectbox("Select Delivery Type", delivery_options)
filtered = df_batsman if delivery == "All" else df_batsman[df_batsman["DeliveryType"] == delivery]

# --------------------------
# Sidebar Metrics
# --------------------------
total_runs = filtered["Runs"].sum()
total_balls = filtered.shape[0]
total_wickets = (filtered["Wicket"] == True).sum()
overall_avg = total_runs / total_wickets if total_wickets > 0 else 0
strike_rate = 100 * total_runs / total_balls if total_balls > 0 else 0

st.sidebar.markdown("### Overall Metrics")
st.sidebar.metric("Strike Rate", f"{strike_rate:.1f}")
st.sidebar.metric("Avg Runs/Wicket", f"{overall_avg:.2f}")
st.sidebar.metric("Total Runs", total_runs)
st.sidebar.metric("Total Balls", total_balls)
st.sidebar.metric("Total Wickets", total_wickets)

# --------------------------
# LAYOUT: Two Columns for Both Charts
# --------------------------
col1, col2 = st.columns([1, 1.2])

# --------------------------
# LEFT CHART: Crease Beehive (Plotly)
# --------------------------
with col1:
    required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ"]
    if not all(col in filtered.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
    else:
        # Separate by wicket
        wickets = filtered[filtered["Wicket"] == True]
        non_wickets = filtered[filtered["Wicket"] == False]

        fig = go.Figure()

        # Non-wickets
        fig.add_trace(go.Scatter(
            x=non_wickets["StumpsY"],
            y=non_wickets["StumpsZ"],
            mode='markers',
            marker=dict(
                color='lightgrey',
                size=8,
                line=dict(width=0),  # No border
                opacity=0.85
            ),
            name="No Wicket"
        ))

        # Wickets
        fig.add_trace(go.Scatter(
            x=wickets["StumpsY"],
            y=wickets["StumpsZ"],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                line=dict(width=0),
                opacity=0.95
            ),
            name="Wicket"
        ))

        # Stumps
        fig.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1.2))
        fig.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1.2))
        fig.add_vline(x=-0.92, line=dict(color="black", width=1))
        fig.add_vline(x=0.92, line=dict(color="black", width=1))

        # Background Zones
        fig.add_shape(type="rect", x0=-2.5, x1=-0.18, y0=0, y1=2.5,
                      fillcolor="rgba(0,255,0,0.05)", line_width=0)
        fig.add_shape(type="rect", x0=0.18, x1=2.5, y0=0, y1=2.5,
                      fillcolor="rgba(255,0,0,0.05)", line_width=0)

        batsman_name = batsman if batsman != "All" else "All Batsmen"
        fig.update_layout(
            title=dict(
                text=f"<b>CBH - {batsman_name}</b>",
                x=0.5,
                y=0.95,
                font=dict(size=20)
            ),
            width=750,
            height=400,
            xaxis=dict(
                range=[-1.6, 1.6],
                showgrid=False,
                zeroline=False,
                visible=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[0, 2.5],
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=False)

# --------------------------
# RIGHT CHART: Zone Heatmap (Matplotlib)
# --------------------------
with col2:
    # Define Zones based on Batting Hand
    right_hand_zones = {
        "Zone 1": (-0.72, 0, -0.45, 1.91),
        "Zone 2": (-0.45, 0, -0.18, 0.71),
        "Zone 3": (-0.18, 0, 0.18, 0.71),
        "Zone 4": (-0.45, 0.71, -0.18, 1.31),
        "Zone 5": (-0.18, 0.71, 0.18, 1.31),
        "Zone 6": (-0.45, 1.31, 0.18, 1.91),
    }

    left_hand_zones = {
        "Zone 1": (0.45, 0, 0.72, 1.91),
        "Zone 2": (0.18, 0, 0.45, 0.71),
        "Zone 3": (-0.18, 0, 0.18, 0.71),
        "Zone 4": (0.18, 0.71, 0.45, 1.31),
        "Zone 5": (-0.18, 0.71, 0.18, 1.31),
        "Zone 6": (-0.18, 1.31, 0.45, 1.91),
    }

    is_right_handed = True
    if batsman != "All":
        handed = filtered["IsBatsmanRightHanded"].dropna().unique()
        is_right_handed = handed[0] if len(handed) > 0 else True

    zones_layout = right_hand_zones if is_right_handed else left_hand_zones

    # Assign Zone
    def assign_zone(row):
        x, y = row["CreaseY"], row["CreaseZ"]
        for zone, (x1, y1, x2, y2) in zones_layout.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone
        return "Other"

    filtered["Zone"] = filtered.apply(assign_zone, axis=1)
    filtered = filtered[filtered["Zone"] != "Other"]

    # Summary by Zone
    summary = (
        filtered.groupby("Zone")
        .agg(
            Runs=("Runs", "sum"),
            Wickets=("Wicket", lambda x: (x == True).sum())
        )
        .reindex(["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"])
        .fillna(0)
    )

    summary["Avg Runs/Wicket"] = summary["Runs"] / summary["Wickets"]
    summary["Avg Runs/Wicket"] = summary["Avg Runs/Wicket"].replace([float("inf"), float("nan")], 0)

    avg_values = summary["Avg Runs/Wicket"]
    norm = mcolors.Normalize(vmin=avg_values.min(), vmax=avg_values.max())
    cmap = cm.get_cmap('Blues')

    fig2, ax = plt.subplots(figsize=(7, 7))

    for zone, (x1, y1, x2, y2) in zones_layout.items():
        w, h = x2 - x1, y2 - y1
        avg = summary.loc[zone, "Avg Runs/Wicket"]
        color = cmap(norm(avg))

        zone_data = filtered[filtered["Zone"] == zone]
        runs = int(summary.loc[zone, "Runs"])
        wkts = int(summary.loc[zone, "Wickets"])
        balls = zone_data.shape[0]
        sr = (runs / balls) * 100 if balls > 0 else 0

        ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=2))
        ax.text(
            x1 + w / 2,
            y1 + h / 2,
            f"{zone}\nRuns: {runs}\nWkts: {wkts}\nAvg: {avg:.1f}\nSR: {sr:.1f}",
            ha="center", va="center", weight="bold", fontsize=10,
            color="black" if norm(avg) < 0.6 else "white"
        )

    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, 2)
    ax.set_xlabel("CreaseY")
    ax.set_ylabel("CreaseZ")
    ax.set_title(batsman if batsman != "All" else "All Batters", fontsize=16)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Avg Runs/Wicket")

    st.pyplot(fig2)
