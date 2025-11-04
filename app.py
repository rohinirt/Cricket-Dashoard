import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# --- 1. SET PAGE CONFIGURATION ---
st.set_page_config(page_title="Cricket Analysis Dashboard")

# --- 2. DATA UPLOADER AND INITIAL LOAD ---
# Use the main area for the file uploader for better visibility
st.title("Cricket Analysis Dashboard 🏏")
st.markdown("Upload your Hawkeye data (CSV) below to begin analysis.")
uploaded_file =uploaded_file = st.file_uploader("Upload your data", type=["csv"])

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

def calculate_scoring_wagon(row):
    """Translates Tableau's trigonometric scoring wagon logic to Python."""
    
    LX = row.get("LandingX")
    LY = row.get("LandingY")
    RH = row.get("IsBatsmanRightHanded")
    
    # Exclude entries where data is missing or no run was scored
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0:
        return None
    
    # Handle division by zero/NaN for ATAN by checking the denominator first
    def atan_safe(numerator, denominator):
        if denominator == 0:
            return np.nan 
        # Use np.arctan for the ATAN function
        return np.arctan(numerator / denominator)

    if RH == True: # Right Handed Batsman
        if LX <= 0 and LY > 0: return "FINE LEG"
        elif LX <= 0 and LY <= 0: return "THIRD MAN"
        
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "COVER"
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG OFF"
        
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "SQUARE LEG"
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG ON"
        
    elif RH == False: # Left Handed Batsman
        if LX <= 0 and LY > 0: return "THIRD MAN"
        elif LX <= 0 and LY <= 0: return "FINE LEG"
            
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "SQUARE LEG"
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG ON"
        
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "COVER"
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG OFF"
                
    return None

def calculate_scoring_angle(area):
    if area in ["FINE LEG", "THIRD MAN"]:
        return 90
    elif area in ["COVER", "SQUARE LEG", "LONG OFF", "LONG ON"]:
        return 45
    return 0 

# Initialize empty summary to prevent errors if processing fails
wagon_summary = pd.DataFrame() 

# ... (Previous code remains, including the definition of calculate_scoring_angle) ...

try:
    df_wagon = filtered_df.copy()
    
    # 1. Calculate Scoring Area
    df_wagon["ScoringWagon"] = df_wagon.apply(calculate_scoring_wagon, axis=1)

    # 2. Calculate Fixed Angle for each row
    df_wagon["FixedAngle"] = df_wagon["ScoringWagon"].apply(calculate_scoring_angle)

    # 3. Summarize Runs and Fixed Angle from the available data
    # Note: We aggregate 'FixedAngle' here, which will be for only the areas with shots
    summary_with_shots = df_wagon.groupby("ScoringWagon").agg(
        TotalRuns=("Runs", "sum"),
        FixedAngle=("FixedAngle", 'first')
    ).reset_index().dropna(subset=["ScoringWagon"])
    
    # Filter for only shots that scored runs for the initial pass (we re-add 0-run areas later)
    # This filter should be removed if you want to include all shots, but keeping it simple for now:
    # summary_with_shots = summary_with_shots[summary_with_shots["TotalRuns"] > 0]


    # --- CRITICAL FIX START: Merging with a Template to Include ALL Areas ---
    
    # Determine handedness and define ALL possible areas
    handedness_mode = filtered_df["IsBatsmanRightHanded"].dropna().mode()
    is_right_handed = handedness_mode.iloc[0] if not handedness_mode.empty else True
    
    if is_right_handed:
        all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"]
    else: # Left Handed
        all_areas = ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]

    # Create a template DataFrame with all 6 required areas and their fixed angles
    template_data = {
        "ScoringWagon": all_areas,
        "FixedAngle": [calculate_scoring_angle(area) for area in all_areas]
    }
    template_df = pd.DataFrame(template_data)

    # Merge the summary with the template to include missing areas with TotalRuns=0
    # We use the FixedAngle from the template as it's guaranteed to be correct for the area.
    wagon_summary = template_df.merge(
        summary_with_shots.drop(columns=["FixedAngle"], errors='ignore'),
        on="ScoringWagon",
        how="left"
    ).fillna(0) # Fill TotalRuns with 0 for areas with no score
    
    # Re-apply the categorical sorting order to the merged DataFrame
    wagon_summary["ScoringWagon"] = pd.Categorical(wagon_summary["ScoringWagon"], categories=all_areas, ordered=True)
    wagon_summary = wagon_summary.sort_values("ScoringWagon").reset_index(drop=True)
    
    # --- CRITICAL FIX END ---


    # 4. Calculate Percentage (This must happen *after* merging all areas)
    total_runs = wagon_summary["TotalRuns"].sum()
    if total_runs > 0:
        wagon_summary["RunPercentage"] = (wagon_summary["TotalRuns"] / total_runs) * 100
    else:
        # If total runs is 0, percentage is 0 for all
        wagon_summary["RunPercentage"] = 0 
        
    # Ensure FixedAngle is numeric for plotting
    wagon_summary["FixedAngle"] = wagon_summary["FixedAngle"].astype(int) 

except KeyError as e:
    # This remains as the overall column error check
    st.error(f"Cannot calculate Wagon Wheel: The required data column {e} is missing. Please ensure your CSV includes 'LandingX' and 'LandingY'.")
    wagon_summary = pd.DataFrame() # Ensure wagon_summary is empty on error


# --- 4. LAYOUT: CHARTS SIDE BY SIDE ---
col1, col2 = st.columns(2)

# ==============================================================================
# CHART 1: CREASE BEEHIVE (In Column 1)
# ==============================================================================
with col1:
    
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
                opacity=0.95
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
                range=[-1.2, 1.2], showgrid=False, zeroline=False, visible=False,
                scaleanchor="y", scaleratio=1
            ),
            yaxis=dict(
                range=[0.5, 2], showgrid=False, zeroline=False, visible=False
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=20, t=60, b=20),
            showlegend=False
        )

        st.plotly_chart(fig_cbh, use_container_width=True)
# ==============================================================================
# CHART 3: PITCH MAP (In Column 1, Bottom - NEW CHART)
# ==============================================================================
# Define Length Bins and Colors
PITCH_BINS = {
    "Short": {"y0": 8.60, "y1": 16.0, "color": "#5d3bb3"},
    "Length": {"y0": 5.0, "y1": 8.60, "color": "#ae4fa1"},
    "Slot": {"y0": 2.8, "y1": 5.0, "color": "#cc5d54"},
    "Yorker": {"y0": 0.9, "y1": 2.8, "color": "#c7b365"}, # Changed Full Toss and Yorker order/colors slightly to match image better
    "Full Toss": {"y0": -4.0, "y1": 0.9, "color": "#6e9d4f"},
}

with col1:
    if filtered_df.empty:
        st.warning("No data matches the selected filters for Pitch Map.")
    else:
        fig_pitch = go.Figure()
        # 1. Add Background Zones (Reversed Y-axis for standard pitch view)
        for length, params in PITCH_BINS.items():
            fig_pitch.add_shape(
                type="rect",
                x0=-1.5, x1=1.5,
                y0=params["y0"], y1=params["y1"],
                fillcolor=params["color"],
                opacity=0.4,
                layer="below",
                line_width=0,
            )
            # Add length labels (Approximate position)
            mid_y = (params["y0"] + params["y1"]/2)
            if length in ["Short", "Length", "Slot", "Yorker", "Full Toss"]:
                fig_pitch.add_annotation(
                    x=-1.45, y=mid_y,
                    text=length.upper(),
                    showarrow=False,
                    font=dict(size=14, color="white",weight='bold'),
                    yref="y",
                    xref="x",
                    xanchor='left'
                )


        # 2. Separate Data by Wicket Status and Plot
        # --- ADDED: Stump lines for Pitch Map ---
        fig_pitch.add_vline(x=-0.18, line=dict(color="#777777", dash="dot", width=1.2))
        fig_pitch.add_vline(x=0.18, line=dict(color="#777777", dash="dot", width=1.2))
        pitch_wickets = filtered_df[filtered_df["Wicket"] == True]
        pitch_non_wickets = filtered_df[filtered_df["Wicket"] == False]

        # Non-wickets (smaller size, white)
        fig_pitch.add_trace(go.Scatter(
            x=pitch_non_wickets["BounceY"], y=pitch_non_wickets["BounceX"],
            mode='markers', name="No Wicket",
            marker=dict(color='white', size=10, line=dict(width=1, color="grey"), opacity=0.9)
        ))

        # Wickets (larger size, color)
        fig_pitch.add_trace(go.Scatter(
            x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"],
            mode='markers', name="Wicket",
            marker=dict(color='red', size=14, line=dict(width=0), opacity=0.95)
        ))

        # 3. Layout Configuration
        fig_pitch.update_layout(
            title=dict(
                text=f"<b>Pitch Map - {batsman_name}</b>", 
                x=0, y=0.95, font=dict(size=20)
            ),
            width=40, 
            height=550, # Increased height for better visualization of lengths
            xaxis=dict(
                range=[-1.5, 1.5],
                showgrid=False, zeroline=False,visible = False
            ),
            yaxis=dict(
                range=[16.0, -4.0], # **Reversed Y-axis**,
                showgrid=False, zeroline=False,visible = False
            ),
            plot_bgcolor="#f6d992", # Setting a default background color for the pitch area
            paper_bgcolor="white",
            margin=dict(l=10, r=20, t=60, b=40),
            showlegend=True
        )

        st.plotly_chart(fig_pitch, use_container_width=True)

# ==============================================================================
# CHART 2: ZONAL BOXES (In Column 2)
# ==============================================================================
with col2:
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
             "Zone 6": (-0.45, 1.31, 0.18, 1.91),
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

        fig_boxes, ax = plt.subplots(figsize=(5, 5)) # Adjusted size

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
        ax.set_xlabel("")  # Hide x-axis label
        ax.set_ylabel("")

        
        handedness = "Right Handed" if is_right_handed else "Left Handed"
        ax.set_title(f"{batsman if batsman != 'All' else 'All Batters'} ({handedness})", fontsize=20)

        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Avg Runs/Wicket")
        
        st.pyplot(fig_boxes)

# ------------------------------------------------------------------------------
# CHART 4: SCORING WAGON WHEEL (In Column 2, Bottom) - FINAL CORRECTED VERSION
# ------------------------------------------------------------------------------
# --- Matplotlib Imports (Ensure these are at the top of your script) ---
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# ----------------------------------------------------------------------

# ... (Previous code remains, including data processing for wagon_summary) ...

# ------------------------------------------------------------------------------
# CHART 4: SCORING WAGON WHEEL (In Column 2, Bottom) - MATPLOTLIB
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# CHART 4: SCORING WAGON WHEEL (In Column 2, Bottom) - MATPLOTLIB (FINAL UPDATED)
# ------------------------------------------------------------------------------
with col2:
    if wagon_summary.empty:
        st.warning("No scoring shots or missing columns prevent the Wagon Wheel from being calculated.")
    else:
        # 1. Prepare Data for Matplotlib (NO CHANGE REQUIRED HERE)
        angles = wagon_summary["FixedAngle"].tolist()
        runs = wagon_summary["TotalRuns"].tolist()
        
        # Labels for outside: Area Name + Run Percentage
        labels = [
            f"{area}\n({pct:.1f}%)" 
            for area, pct in zip(wagon_summary["ScoringWagon"], wagon_summary["RunPercentage"])
        ]
        
        # Calculate run percentages for custom label function (NO CHANGE REQUIRED HERE)
        total_runs = sum(runs)
        percentages = [(r / total_runs) * 100 if total_runs > 0 else 0 for r in runs]
        
        # NOTE: The custom autopct_format function is NO LONGER NEEDED as we calculate
        # the full labels list directly above, but I'll keep the definition block
        # below to maintain structure consistency if you are using it elsewhere.
        def autopct_format(pct):
             # This function is now only used as a placeholder in ax.pie
             # to trigger the counter if needed, but we don't use its output.
             return '' 

        # 2. Setup Coloring (Heatmap Effect)
        run_min = min(runs)
        run_max = max(runs)
        
        # Create a Normalization object based on runs (NO CHANGE REQUIRED HERE)
        if run_max > run_min:
            norm = mcolors.Normalize(vmin=run_min, vmax=run_max)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)
        
        # Choose the Colormap (I'll switch back to 'Reds' as 'Blues' was used in the Zonal Chart)
        cmap = cm.get_cmap('Blues')
        
        # Map the run totals to colors
        colors = cmap(norm(runs))
        
        # --- FIX FOR WHITE 0% SLICES (Confirms the logic) ---
        # Set slices with 0 runs to white
        for i, run_count in enumerate(runs):
            if run_count == 0:
                colors[i] = (1.0, 1.0, 1.0, 1.0)

        # 3. Create the Matplotlib Figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # The 'angles' array controls the size of the slices, which are fixed (90 or 45)
        # Use the 'labels' argument for outside text, and 'pctdistance' to move the text
        # 'autopct' is set to a dummy value (like a fixed empty string) or removed,
        # but since we want the text *outside*, we use 'labels'.
        wedges, texts = ax.pie(
            angles, 
            colors=colors, 
            wedgeprops={"width": 1, "edgecolor": "black"}, # Full pie width
            startangle=90, 
            counterclock=False, 
            labels=labels, # --- NEW: Use labels argument for outside text ---
            labeldistance=1.1, # Push labels outside
            # pctdistance is not needed since we are not using the inner text (autopct)
        )
        
        # 4. Customize Text Properties (Outside Labels)
        for text in texts:
            text.set_color('black')
            text.set_fontsize(12)
            text.set_fontweight('bold')

        # --- REMOVE MANUAL TEXT PLACEMENT LOOP ---
        # The entire loop that manually calculated x,y and used ax.text() is removed.
        # -----------------------------------------

        ax.set_title(f"Scoring Areas - {batsman if batsman != 'All' else 'All Batters'}", fontsize=22, fontweight='bold')
        ax.axis('equal') 

        st.pyplot(fig)
