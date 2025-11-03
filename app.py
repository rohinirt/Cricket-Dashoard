import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- 1. SET PAGE CONFIGURATION ---
st.set_page_config(page_title="Cricket Analysis Dashboard")

# --- 2. DATA UPLOADER AND INITIAL LOAD ---
# Use the main area for the file uploader for better visibility
st.title("Cricket Analysis Dashboard 🏏")
st.markdown("Upload your Hawkeye data (CSV) below to begin analysis.")
uploaded_file = st.file_uploader("Upload your data", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Required columns check (adjusting for both charts)
        required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing one or more required columns. Required: {', '.join(required_cols)}")
            df = None
        else:
            st.success("File uploaded and validated successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")

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

# --- 4. LAYOUT: CHARTS SIDE BY SIDE ---
col1, col2 = st.columns(2)

# ==============================================================================
# CHART 1: CREASE BEEHIVE (In Column 1)
# ==============================================================================
with col1:
    st.header("Crease Beehive (CBH)")
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters for CBH.")
    else:
        # --- Separate by wicket ---
        wickets = filtered_df[filtered_df["Wicket"] == True]
        non_wickets = filtered_df[filtered_df["Wicket"] == False]

        # --- Create Plotly figure ---
        fig_cbh = go.Figure()

        # Non-wickets (grey with no border)
        fig_cbh.add_trace(go.Scatter(
            x=non_wickets["StumpsY"],
            y=non_wickets["StumpsZ"],
            mode='markers',
            marker=dict(
                color='lightgrey',
                size=8,
                line=dict(width=0),
                opacity=0.85
            ),
            name="No Wicket"
        ))

        # Wickets (red with no border)
        fig_cbh.add_trace(go.Scatter(
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

        # --- Stump lines & Background zones ---
        fig_cbh.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1.2))
        fig_cbh.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1.2))
        fig_cbh.add_vline(x=-0.92, line=dict(color="black", width=1))
        fig_cbh.add_vline(x=0.92, line=dict(color="black", width=1))

        # Background zones (left/right of middle stump)
        fig_cbh.add_shape(type="rect", x0=-2.5, x1=-0.18, y0=0, y1=2.5,
                          fillcolor="rgba(0,255,0,0.05)", line_width=0)
        fig_cbh.add_shape(type="rect", x0=0.18, x1=2.5, y0=0, y1=2.5,
                          fillcolor="rgba(255,0,0,0.05)", line_width=0)

        # --- Chart Layout ---
        batsman_name = batsman if batsman != "All" else "All Batsmen"
        fig_cbh.update_layout(
            title=dict(
                text=f"<b>CBH - {batsman_name}</b>",
                x=0, y=0.95, font=dict(size=20)
            ),
            width=700, # Adjusted width for side-by-side view
            height=400,
            xaxis=dict(
                range=[-1.2, 1.2], showgrid=False, zeroline=False, visible=True,
                scaleanchor="y", scaleratio=1
            ),
            yaxis=dict(
                range=[0.5, 2], showgrid=False, zeroline=False, visible=True
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )

        st.plotly_chart(fig_cbh, use_container_width=True)

# ==============================================================================
# CHART 2: ZONAL BOXES (In Column 2)
# ==============================================================================
with col2:
    st.header("Crease Zonal Analysis (Avg Runs/Wicket)")

    if filtered_df.empty:
        st.warning("No data matches the selected filters for Zonal Analysis.")
    else:
        # --- Define Zones based on Batting Hand ---
        right_hand_zones = {
            "Zone 1": (-0.72, 0, -0.45, 1.91),
            "Zone 2": (-0.45, 0, -0.18, 0.71),
            "Zone 3": (-0.18, 0, 0.18, 0.71),
            "Zone 4": (-0.45, 0.71, -0.18, 1.31),
            "Zone 5": (-0.18, 0.71, 0.18, 1.31),
             "Zone 6": (-0.45, 1.31, 0.18, 1.91), # Note: Zone 6 seems defined differently in original
        }

        left_hand_zones = {
            "Zone 1": (0.45, 0, 0.72, 1.91),
            "Zone 2": (0.18, 0, 0.45, 0.71),
            "Zone 3": (-0.18, 0, 0.18, 0.71),
            "Zone 4": (0.18, 0.71, 0.45, 1.31),
            "Zone 5": (-0.18, 0.71, 0.18, 1.31),
            "Zone 6": (-0.18, 1.31, 0.45, 1.91), # Note: Zone 6 seems defined differently in original
        }

        # Detect handedness (Default to Right Hand if data is ambiguous/missing)
        is_right_handed = True
        handed_data = filtered_df["IsBatsmanRightHanded"].dropna().unique()
        if len(handed_data) > 0 and batsman != "All":
            is_right_handed = handed_data[0]
        
        zones_layout = right_hand_zones if is_right_handed else left_hand_zones
        
        # --- Assign Zone Function ---
        def assign_zone(row):
            x, y = row["CreaseY"], row["CreaseZ"]
            for zone, (x1, y1, x2, y2) in zones_layout.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return zone
            return "Other"

        # Apply assignment to a copy to avoid SettingWithCopyWarning
        df_chart2 = filtered_df.copy()
        df_chart2["Zone"] = df_chart2.apply(assign_zone, axis=1)
        df_chart2 = df_chart2[df_chart2["Zone"] != "Other"]

        # --- Calculate Summary ---
        summary = (
            df_chart2.groupby("Zone")
            .agg(
                Runs=("Runs", "sum"),
                Wickets=("Wicket", lambda x: (x == True).sum()),
                Balls=("Wicket", "count")
            )
            .reindex(["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"])
            .fillna(0)
        )

        # Avoid division by zero/inf
        summary["Avg Runs/Wicket"] = summary.apply(
            lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0,
            axis=1
        )
        summary["StrikeRate"] = summary.apply(
            lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else 0,
            axis=1
        )

        # --- Heatmap Plotting (Matplotlib) ---
        avg_values = summary["Avg Runs/Wicket"]
        if avg_values.empty or avg_values.max() == 0:
             # Handle case where all avg runs are zero or no data
             norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            norm = mcolors.Normalize(vmin=avg_values[avg_values > 0].min(), vmax=avg_values.max())
        
        cmap = cm.get_cmap('Blues')

        fig_boxes, ax = plt.subplots(figsize=(7, 7)) # Adjusted size

        for zone, (x1, y1, x2, y2) in zones_layout.items():
            w, h = x2 - x1, y2 - y1
            
            if zone not in summary.index:
                runs, wkts, avg, sr = 0, 0, 0, 0
                color = 'white' # No data color
            else:
                runs = int(summary.loc[zone, "Runs"])
                wkts = int(summary.loc[zone, "Wickets"])
                avg = summary.loc[zone, "Avg Runs/Wicket"]
                sr = summary.loc[zone, "StrikeRate"]
                color = cmap(norm(avg))

            ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=2))

            ax.text(
                x1 + w / 2, y1 + h / 2,
                f"{zone}\nRuns: {runs}\nWkts: {wkts}\nAvg: {avg:.1f}\nSR: {sr:.1f}",
                ha="center", va="center", weight="bold", fontsize=9,
                color="black" if norm(avg) < 0.6 else "white"
            )

        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(0, 2)
        ax.set_xlabel("CreaseY (Distance from Middle Stump)")
        ax.set_ylabel("CreaseZ (Height from Ground)")
        
        handedness = "Right Handed" if is_right_handed else "Left Handed"
        ax.set_title(f"{batsman if batsman != 'All' else 'All Batters'} ({handedness})", fontsize=14)

        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Avg Runs/Wicket")

        st.pyplot(fig_boxes)
