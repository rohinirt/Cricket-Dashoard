import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from io import StringIO
import base64

# --- 1. GLOBAL UTILITY FUNCTIONS ---

# Required columns check
REQUIRED_COLS = [
    "BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", 
    "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", 
    "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", 
    "InterceptionZ", "InterceptionY", "Over"
]

# Wagon Wheel Calculation (Function remains the same)
def calculate_scoring_wagon(row):
    LX = row.get("LandingX"); LY = row.get("LandingY"); RH = row.get("IsBatsmanRightHanded")
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0: return None
    def atan_safe(numerator, denominator): return np.arctan(numerator / denominator) if denominator != 0 else np.nan 
    
    # Right Handed Batsman Logic
    if RH == True: 
        if LX <= 0 and LY > 0: return "FINE LEG"
        elif LX <= 0 and LY <= 0: return "THIRD MAN"
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "COVER"
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG OFF"
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "SQUARE LEG"
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG ON"
    # Left Handed Batsman Logic
    elif RH == False: 
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
    if area in ["FINE LEG", "THIRD MAN"]: return 90
    elif area in ["COVER", "SQUARE LEG", "LONG OFF", "LONG ON"]: return 45
    return 0

# Function to encode Matplotlib figure to image for Streamlit
def fig_to_image(fig):
    return fig

# --- 2. CHART GENERATION FUNCTIONS (REMAINS THE SAME, BUT INTERCEPTION LOGIC IS NOW CORRECTED) ---

# --- CHART 1: ZONAL ANALYSIS (CBH Boxes) ---
def create_zonal_analysis(df_in, batsman_name, delivery_type):
    # ... (Zonal Analysis logic remains the same)
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig

    is_right_handed = True
    handed_data = df_in["IsBatsmanRightHanded"].dropna().unique()
    if len(handed_data) > 0 and batsman_name != "All": is_right_handed = handed_data[0]
        
    right_hand_zones = { "Z1": (-0.72, 0, -0.45, 1.91), "Z2": (-0.45, 0, -0.18, 0.71), "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (-0.45, 0.71, -0.18, 1.31), "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.45, 1.31, 0.18, 1.91)}
    left_hand_zones = { "Z1": (0.45, 0, 0.72, 1.91), "Z2": (0.18, 0, 0.45, 0.71), "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (0.18, 0.71, 0.45, 1.31), "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.18, 1.31, 0.45, 1.91)}
    zones_layout = right_hand_zones if is_right_handed else left_hand_zones
    
    def assign_zone(row):
        x, y = row["CreaseY"], row["CreaseZ"]
        for zone, (x1, y1, x2, y2) in zones_layout.items():
            if x1 <= x <= x2 and y1 <= y <= y2: return zone
        return "Other"

    df_chart2 = df_in.copy(); df_chart2["Zone"] = df_chart2.apply(assign_zone, axis=1)
    df_chart2 = df_chart2[df_chart2["Zone"] != "Other"]
    
    summary = (
        df_chart2.groupby("Zone").agg(Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: (x == True).sum()), Balls=("Wicket", "count"))
        .reindex([f"Z{i}" for i in range(1, 7)]).fillna(0)
    )
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)
    summary["StrikeRate"] = summary.apply(lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else 0, axis=1)

    avg_values = summary["Avg Runs/Wicket"]
    avg_max = avg_values.max() if avg_values.max() > 0 else 1
    avg_min = avg_values[avg_values > 0].min() if avg_values[avg_values > 0].min() < avg_max else 0
    norm = mcolors.Normalize(vmin=avg_min, vmax=avg_max)
    cmap = cm.get_cmap('Blues')

    fig_boxes, ax = plt.subplots(figsize=(4, 4), subplot_kw={'xticks': [], 'yticks': []}) 
    
    for zone, (x1, y1, x2, y2) in zones_layout.items():
        w, h = x2 - x1, y2 - y1
        z_key = zone.replace("Zone ", "Z")
        
        runs, wkts, avg, sr = (0, 0, 0, 0)
        if z_key in summary.index:
            runs = int(summary.loc[z_key, "Runs"])
            wkts = int(summary.loc[z_key, "Wickets"])
            avg = summary.loc[z_key, "Avg Runs/Wicket"]
            sr = summary.loc[z_key, "StrikeRate"]
        
        color = cmap(norm(avg)) if avg > 0 else 'white'

        ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=1.5))

        ax.text(x1 + w / 2, y1 + h / 2, 
                f"{z_key}\nR:{runs} W:{wkts}\nA:{avg:.1f} SR:{sr:.1f}", 
                ha="center", va="center", weight="bold", fontsize=7,
                color="black" if norm(avg) < 0.6 else "white", 
                linespacing=1.2)

    ax.set_xlim(-0.75, 0.75); ax.set_ylim(0, 2); ax.axis('off'); 
    ax.set_title(f"Zonal Analysis ({delivery_type})", fontsize=10, weight='bold')
    plt.tight_layout(pad=0.5) 
    return fig_boxes

# --- CHART 2: CREASE BEEHIVE ---
def create_crease_beehive(df_in, delivery_type):
    # ... (Crease Beehive logic remains the same)
    if df_in.empty:
        return go.Figure().update_layout(title="No data for Beehive", height=300)

    wickets = df_in[df_in["Wicket"] == True]
    non_wickets = df_in[df_in["Wicket"] == False]
    fig_cbh = go.Figure()

    fig_cbh.add_trace(go.Scatter(
        x=non_wickets["StumpsY"], y=non_wickets["StumpsZ"], mode='markers', name="No Wicket",
        marker=dict(color='lightgrey', size=4, line=dict(width=0), opacity=0.95)
    ))

    fig_cbh.add_trace(go.Scatter(
        x=wickets["StumpsY"], y=wickets["StumpsZ"], mode='markers', name="Wicket",
        marker=dict(color='red', size=8, line=dict(width=0), opacity=0.95)
    ))

    # Stump lines & Crease lines
    fig_cbh.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1)) 
    fig_cbh.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1))
    fig_cbh.add_vline(x=-0.92, line=dict(color="grey", width=0.8)) 
    fig_cbh.add_vline(x=0.92, line=dict(color="grey", width=0.8))
    fig_cbh.add_hline(y=0.78, line=dict(color="grey", width=0.8)) 
    
    fig_cbh.update_layout(
        title=f"Crease Beehive ({delivery_type})",
        height=300, 
        margin=dict(l=0, r=0, t=30, b=10),
        xaxis=dict(range=[-1.6, 1.6], showgrid=True, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0.5, 2], showgrid=True, zeroline=False, visible=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    return fig_cbh


# --- CHART 3: PITCH MAP ---
def create_pitch_map(df_in, delivery_type):
    # ... (Pitch Map logic remains the same)
    if df_in.empty:
        return go.Figure().update_layout(title="No data for Pitch Map", height=350)

    PITCH_BINS = {
        "Short": {"y0": 8.60, "y1": 16.0},
        "Length": {"y0": 5.0, "y1": 8.60},
        "Slot": {"y0": 2.8, "y1": 5.0},
        "Yorker": {"y0": 0.9, "y1": 2.8},
        "Full Toss": {"y0": -4.0, "y1": 0.9},
    }
    
    fig_pitch = go.Figure()
    
    # 1. Add Zone Lines & Labels
    for length, params in PITCH_BINS.items():
        fig_pitch.add_hline(y=params["y0"], line=dict(color="lightgrey", width=1.0, dash="dot"))
        mid_y = (params["y0"] + params["y1"]) / 2
        fig_pitch.add_annotation(x=-1.45, y=mid_y, text=length.upper(), showarrow=False,
            font=dict(size=8, color="grey", weight='bold'), xanchor='left')

    # 2. Add Stump lines
    fig_pitch.add_vline(x=-0.18, line=dict(color="#777777", dash="dot", width=1.2))
    fig_pitch.add_vline(x=0.18, line=dict(color="#777777", dash="dot", width=1.2))

    # 3. Plot Data
    pitch_wickets = df_in[df_in["Wicket"] == True]
    pitch_non_wickets = df_in[df_in["Wicket"] == False]

    fig_pitch.add_trace(go.Scatter(
        x=pitch_non_wickets["BounceY"], y=pitch_non_wickets["BounceX"], mode='markers', name="No Wicket",
        marker=dict(color='white', size=4, line=dict(width=1, color="grey"), opacity=0.9)
    ))

    fig_pitch.add_trace(go.Scatter(
        x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"], mode='markers', name="Wicket",
        marker=dict(color='red', size=8, line=dict(width=0), opacity=0.95)
    ))

    # 4. Layout
    fig_pitch.update_layout(
        title=f"Pitch Map (Bounce Location - {delivery_type})",
        height=350, 
        margin=dict(l=0, r=0, t=30, b=10),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[16.0, -4.0], showgrid=False, zeroline=False, visible=False), 
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    return fig_pitch

# --- CHART 4: INTERCEPTION SIDE-ON --- (Wide View)
def create_interception_side_on(df_in, delivery_type):
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()
    if df_interception.empty:
        fig, ax = plt.subplots(figsize=(3, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
        
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    # Define color_map inline as it's needed for the loop
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    fig_7, ax_7 = plt.subplots(figsize=(3, 4), subplot_kw={'xticks': [], 'yticks': []}) 
    
    # 1. Plot Data (Layered for correct border visibility)
    
    # Plot "Other" (White with Grey Border)
    df_other = df_interception[df_interception["ColorType"] == "Other"]
    # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on X-axis ===
    ax_7.scatter(
        df_other["InterceptionX"] + 10, df_other["InterceptionZ"], 
        color='white', edgecolors='grey', linewidths=0.5, s=40, label="Other"
    )
    
    # Plot "Wicket" and "Boundary" (Solid colors)
    for ctype in ["Boundary", "Wicket"]:
        df_slice = df_interception[df_interception["ColorType"] == ctype]
        # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on X-axis ===
        ax_7.scatter(
            df_slice["InterceptionX"] + 10, df_slice["InterceptionZ"], 
            color=color_map[ctype], s=40, label=ctype
        )

    # 2. Draw Vertical Dashed Lines with Labels (FIXED LINES: 0.0, 1.25, 2.0, 3.0)
    line_specs = {
        0.0: "Stumps",
        1.250: "Crease",
        2.000: "2m",     
        3.000: "3m"      
    }
    
    for x_val, label in line_specs.items():
        ax_7.axvline(x=x_val, color='grey', linestyle='--', linewidth=1, alpha=0.7)    
        ax_7.text(x_val, 1.45, label.split(':')[-1].strip(), ha='center', va='center', fontsize=6, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 3. Set Axes Limits and Labels (FIXED LIMITS: -0.2 to 3.4)
    ax_7.set_xlim(-0.2, 3.4) 
    ax_7.set_ylim(0, 1.5) 
    ax_7.tick_params(axis='y', which='both', labelleft=False, left=False); ax_7.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_7.set_xlabel("Distance (m)", fontsize=8); ax_7.set_ylabel("Height (m)", fontsize=8) 
    ax_7.legend(loc='upper right', fontsize=6); ax_7.grid(True, linestyle=':', alpha=0.5); 
    ax_7.set_title(f"Interception Side-On ({delivery_type})", fontsize=10, weight='bold')
    plt.tight_layout(pad=0.5)
    return fig_7


# --- CHART 5: INTERCEPTION FRONT-ON --- (Distance vs Width)
def create_interception_front_on(df_in, delivery_type):
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()
    if df_interception.empty:
        fig, ax = plt.subplots(figsize=(3, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
        
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    # Define color_map inline as it's needed for the loop
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    fig_8, ax_8 = plt.subplots(figsize=(3, 4), subplot_kw={'xticks': [], 'yticks': []}) 

    # 1. Plot Data
    # Plot "Other" (White with Grey Border)
    df_other = df_interception[df_interception["ColorType"] == "Other"]
    # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on Y-axis (Distance) ===
    ax_8.scatter(
        df_other["InterceptionY"], df_other["InterceptionX"] + 10, 
        color='white', edgecolors='grey', linewidths=0.5, s=40, label="Other"
    ) 
    
    # Plot "Wicket" and "Boundary" (Solid colors)
    for ctype in ["Boundary", "Wicket"]:
        df_slice = df_interception[df_interception["ColorType"] == ctype]
        # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on Y-axis (Distance) ===
        ax_8.scatter(
            df_slice["InterceptionY"], df_slice["InterceptionX"] + 10, 
            color=color_map[ctype], s=40, label=ctype
        ) 

    # 2. Draw Horizontal Dashed Lines with Labels (FIXED LINES: 0.0, 1.25)
    line_specs = {
        0.00: "Stumps",
        1.25: "Crease"        
    }
    for y_val, label in line_specs.items():
        ax_8.axhline(y=y_val, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        ax_8.text(-0.95, y_val, label.split(':')[-1].strip(), ha='left', va='center', fontsize=6, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Boundary lines (FIXED LINES: -0.18, 0.18)
    ax_8.axvline(x=-0.18, color='grey', linestyle='-', linewidth=1.5, alpha=0.7)
    ax_8.axvline(x= 0.18, color='grey', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # 3. Set Axes Limits and Labels (FIXED LIMITS: Y-axis -0.2 to 3.5)
    ax_8.set_xlim(-1, 1); ax_8.set_ylim(-0.2, 3.5); ax_8.invert_yaxis()      
    ax_8.tick_params(axis='y', which='both', labelleft=False, left=False); ax_8.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_8.set_xlabel("Width (m)", fontsize=8); ax_8.set_ylabel("Distance (m)", fontsize=8) 
    ax_8.legend(loc='lower right', fontsize=6); ax_8.grid(True, linestyle=':', alpha=0.5); 
    ax_8.set_title(f"Interception Front-On ({delivery_type})", fontsize=10, weight='bold')
    plt.tight_layout(pad=0.5)
    return fig_8


# --- CHART 6: SCORING WAGON WHEEL ---
def create_wagon_wheel(df_in, delivery_type):
    # ... (Wagon Wheel logic remains the same)
    wagon_summary = pd.DataFrame() 
    try:
        df_wagon = df_in.copy()
        df_wagon["ScoringWagon"] = df_wagon.apply(calculate_scoring_wagon, axis=1)
        df_wagon["FixedAngle"] = df_wagon["ScoringWagon"].apply(calculate_scoring_angle)
        
        summary_with_shots = df_wagon.groupby("ScoringWagon").agg(TotalRuns=("Runs", "sum"), FixedAngle=("FixedAngle", 'first')).reset_index().dropna(subset=["ScoringWagon"])
        handedness_mode = df_in["IsBatsmanRightHanded"].dropna().mode()
        is_right_handed = handedness_mode.iloc[0] if not handedness_mode.empty else True
        
        if is_right_handed:
            all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"] 
        else:
            all_areas = ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]
            
        template_df = pd.DataFrame({"ScoringWagon": all_areas, "FixedAngle": [calculate_scoring_angle(area) for area in all_areas]})

        wagon_summary = template_df.merge(summary_with_shots.drop(columns=["FixedAngle"], errors='ignore'), on="ScoringWagon", how="left").fillna(0) 
        wagon_summary["ScoringWagon"] = pd.Categorical(wagon_summary["ScoringWagon"], categories=all_areas, ordered=True)
        wagon_summary = wagon_summary.sort_values("ScoringWagon").reset_index(drop=True)
        
        total_runs = wagon_summary["TotalRuns"].sum()
        wagon_summary["RunPercentage"] = (wagon_summary["TotalRuns"] / total_runs) * 100 if total_runs > 0 else 0 
        wagon_summary["FixedAngle"] = wagon_summary["FixedAngle"].astype(int) 
    except Exception:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Scoring Data", ha='center', va='center'); ax.axis('off'); return fig


    angles = wagon_summary["FixedAngle"].tolist()
    runs = wagon_summary["TotalRuns"].tolist()
    labels = [f"{area}\n({pct:.0f}%)" for area, pct in zip(wagon_summary["ScoringWagon"], wagon_summary["RunPercentage"])]
    
    run_min = min(runs) if runs else 0
    run_max = max(runs) if runs else 1
    norm = mcolors.Normalize(vmin=run_min, vmax=run_max) if run_max > run_min else mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Greens')
    colors = cmap(norm(runs))
    for i, run_count in enumerate(runs):
        if run_count == 0: colors[i] = (1.0, 1.0, 1.0, 1.0) 

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'xticks': [], 'yticks': []}) 
    
    wedges, texts = ax.pie(
        angles, 
        colors=colors, 
        wedgeprops={"width": 1, "edgecolor": "black"}, 
        startangle=90, 
        counterclock=False, 
        labels=labels, 
        labeldistance=1.1
    )
    
    for text in texts:
        text.set_color('black'); text.set_fontsize(8); text.set_fontweight('bold')

    ax.axis('equal'); 
    ax.set_title(f"Scoring Areas ({delivery_type})", fontsize=10, weight='bold')
    plt.tight_layout(pad=0.5)
    
    return fig


# --- 3. MAIN STREAMLIT APP STRUCTURE ---

st.set_page_config(layout="wide")

st.title("🏏 Cricket Bowling Delivery Analysis Dashboard")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file is not None:
    # Read the data from the uploaded file
    try:
        data = uploaded_file.getvalue().decode("utf-8")
        df_raw = pd.read_csv(StringIO(data))
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    # Initial validation and required column check
    if not all(col in df_raw.columns for col in REQUIRED_COLS):
        missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
        st.error(f"The CSV file is missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # Data separation
    df_seam = df_raw[df_raw["DeliveryType"] == "Seam"].copy()
    df_spin = df_raw[df_raw["DeliveryType"] == "Spin"].copy()

    # --- Global Filters (Apply to both Seam and Spin data) ---
    st.sidebar.header("Global Filters")
    
    all_teams = ["All"] + sorted(df_raw["BattingTeam"].dropna().unique().tolist())
    bat_team = st.sidebar.selectbox("Batting Team", all_teams, index=0)

    if bat_team != "All":
        batsmen_options = ["All"] + sorted(df_raw[df_raw["BattingTeam"] == bat_team]["BatsmanName"].dropna().unique().tolist())
    else:
        batsmen_options = ["All"] + sorted(df_raw["BatsmanName"].dropna().unique().tolist())
        
    batsman = st.sidebar.selectbox("Batsman Name", batsmen_options, index=0)

    all_overs = ["All"] + sorted(df_raw["Over"].dropna().unique().tolist())
    selected_over = st.sidebar.selectbox("Over", all_overs, index=0)

    # --- Apply Filters to Seam and Spin dataframes ---
    def apply_filters(df):
        if bat_team != "All":
            df = df[df["BattingTeam"] == bat_team]
        if batsman != "All":
            df = df[df["BatsmanName"] == batsman]
        if selected_over != "All":
            df = df[df["Over"] == selected_over]
        return df

    df_seam = apply_filters(df_seam)
    df_spin = apply_filters(df_spin)
    
    heading_text = batsman.upper() if batsman != "All" else "GLOBAL ANALYSIS"
    st.header(f"Analysis for: **{heading_text}**")
    st.markdown("---")


    # --- 4. DISPLAY CHARTS IN TWO COLUMNS ---
    
    col1, col2 = st.columns(2)
    
    # --- LEFT COLUMN: SEAM ANALYSIS ---
    with col1:
        st.subheader("LEFT COLUMN: SEAM ANALYSIS 🥎")
        st.markdown("---")

        st.markdown("##### 1. Zonal Analysis (Crease Impact)")
        st.pyplot(create_zonal_analysis(df_seam, batsman, "Seam"), use_container_width=True)

        st.markdown("##### 2. Crease Beehive")
        st.plotly_chart(create_crease_beehive(df_seam, "Seam"), use_container_width=True)

        st.markdown("##### 3. Pitch Map (Bounce Location)")
        st.plotly_chart(create_pitch_map(df_seam, "Seam"), use_container_width=True)
        
        # --- NEW LAYOUT START ---
        
        # Chart 4: Interception Side-On (Wide View) - Takes full width
        st.markdown("##### 4. Interception Side-On (Wide View)")
        st.pyplot(create_interception_side_on(df_seam, "Seam"), use_container_width=True)

        # Charts 5 & 6: Interception Front-On and Scoring Areas (Side-by-Side)
        bottom_col_left, bottom_col_right = st.columns(2)

        with bottom_col_left:
            st.markdown("##### 5. Interception Front-On")
            st.pyplot(create_interception_front_on(df_seam, "Seam"), use_container_width=True)
        
        with bottom_col_right:
            st.markdown("##### 6. Scoring Areas (Wagon Wheel)")
            st.pyplot(create_wagon_wheel(df_seam, "Seam"), use_container_width=True)
            
        # --- NEW LAYOUT END ---


    # --- RIGHT COLUMN: SPIN ANALYSIS ---
    with col2:
        st.subheader("RIGHT COLUMN: SPIN ANALYSIS 🌀")
        st.markdown("---")
        
        st.pyplot(create_zonal_analysis(df_spin, batsman, "Spin"), use_container_width=True)
        
        st.plotly_chart(create_crease_beehive(df_spin, "Spin"), use_container_width=True)

        st.plotly_chart(create_pitch_map(df_spin, "Spin"), use_container_width=True)
        
        # --- NEW LAYOUT START (Mirroring Left Column) ---
        
        # Chart 4: Interception Side-On (Wide View) - Takes full width
        st.pyplot(create_interception_side_on(df_spin, "Spin"), use_container_width=True)

        # Charts 5 & 6: Interception Front-On and Scoring Areas (Side-by-Side)
        bottom_col_left, bottom_col_right = st.columns(2)

        with bottom_col_left:
            st.pyplot(create_interception_front_on(df_spin, "Spin"), use_container_width=True)
        
        with bottom_col_right:
            st.pyplot(create_wagon_wheel(df_spin, "Spin"), use_container_width=True)
            
        # --- NEW LAYOUT END ---

else:
    st.info("⬆️ Please upload a CSV file to begin the analysis.")
