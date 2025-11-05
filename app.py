import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# --- 1. SET PAGE CONFIGURATION ---
st.set_page_config(page_title="Cricket Analysis Dashboard", layout="wide")

# --- 2. INJECT CUSTOM CSS FOR ABSOLUTE MINIMUM SPACING ---
st.markdown("""
<style>
    /* 1. Target the main content container and set absolute minimum padding */
    .block-container {
        padding-top: 0.1rem; 
        padding-right: 0.1rem;
        padding-left: 0.1rem;
        padding-bottom: 0.1rem;
    }
    
    /* 2. Target the main section and reduce vertical margins */
    section.main, .css-18e3th9 {
        padding-top: 0.1rem; 
        padding-bottom: 0.1rem;
    }
    
    /* 3. Reduce vertical space around markdown text/headers/captions */
    h1, h2, h3, h4, p, .stCaption {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* 4. Remove padding from columns for tight chart spacing */
    .st-emotion-cache-1om0885 { 
        padding: 0px !important; 
    }

    /* 5. Reduce spacing for the separator line */
    hr {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. DATA UPLOADER AND INITIAL LOAD ---
st.title("Cricket Analysis Dashboard 🏏")
uploaded_file = st.file_uploader("Upload your data", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", "InterceptionZ", "InterceptionY", "Over"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing one or more required columns. Required: {', '.join(required_cols)}. Please ensure your columns are named correctly.")
            df = None
        else:
            st.success("File uploaded and validated successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")

if df is None:
    st.info("👆 Please upload a CSV file with the required Hawkeye data columns to view the dashboard.")
    st.stop() 

# --- 4. TOP FILTERS (COMPACT ROW) ---
filter_col1, filter_col2 = st.columns([1.5, 1.5])

# Filter 1 & 2: Batting Team & Batsman Name
with filter_col1:
    bat_team_options = ["All"] + sorted(df["BattingTeam"].dropna().unique().tolist())
    bat_team = st.selectbox("Batting Team", bat_team_options, label_visibility="collapsed")
    df_bat_team = df if bat_team == "All" else df[df["BattingTeam"] == bat_team]
    st.caption(f"**Team:** {bat_team}") 
    
    batsman_options = ["All"] + sorted(df_bat_team["BatsmanName"].dropna().unique().tolist())
    batsman = st.selectbox("Batsman", batsman_options, label_visibility="collapsed")
    df_batsman = df_bat_team if batsman == "All" else df_bat_team[df_bat_team["BatsmanName"] == batsman]
    st.caption(f"**Batsman:** {batsman}") 

# Filter 3: Over
with filter_col2:
    over_options = ["All"] + sorted(df_batsman["Over"].dropna().unique().tolist())
    selected_over = st.selectbox("Over", over_options, label_visibility="collapsed")
    df_over = df_batsman if selected_over == "All" else df_batsman[df_batsman["Over"] == selected_over]
    st.caption(f"**Over:** {selected_over}") 
    
    # Static info display
    st.caption(f"**Analysis Filter:** Seam")

# --- FIXED FILTER: DELIVERY TYPE = 'Seam' ---
filtered_df = df_over[df_over["DeliveryType"] == "Seam"]

st.markdown("---")

# --- WAGON WHEEL UTILITY FUNCTIONS (Retained) ---
def calculate_scoring_wagon(row):
    # ... [Function body remains the same] ...
    LX = row.get("LandingX")
    LY = row.get("LandingY")
    RH = row.get("IsBatsmanRightHanded")
    
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0:
        return None
    
    def atan_safe(numerator, denominator):
        if denominator == 0:
            return np.nan 
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
            elif atan_safe(LX, LY) <= np.pi / 4: return "LONG OFF"
                
    return None

def calculate_scoring_angle(area):
    if area in ["FINE LEG", "THIRD MAN"]:
        return 90
    elif area in ["COVER", "SQUARE LEG", "LONG OFF", "LONG ON"]:
        return 45
    return 0 

# --- WAGON WHEEL DATA PROCESSING (Retained) ---
wagon_summary = pd.DataFrame() 
try:
    df_wagon = filtered_df.copy()
    df_wagon["ScoringWagon"] = df_wagon.apply(calculate_scoring_wagon, axis=1)
    df_wagon["FixedAngle"] = df_wagon["ScoringWagon"].apply(calculate_scoring_angle)
    
    summary_with_shots = df_wagon.groupby("ScoringWagon").agg(
        TotalRuns=("Runs", "sum"),
        FixedAngle=("FixedAngle", 'first')
    ).reset_index().dropna(subset=["ScoringWagon"])
    
    handedness_mode = filtered_df["IsBatsmanRightHanded"].dropna().mode()
    is_right_handed = handedness_mode.iloc[0] if not handedness_mode.empty else True
    
    if is_right_handed:
        all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"]
    else: 
        all_areas = ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]

    template_data = {
        "ScoringWagon": all_areas,
        "FixedAngle": [calculate_scoring_angle(area) for area in all_areas]
    }
    template_df = pd.DataFrame(template_data)

    wagon_summary = template_df.merge(
        summary_with_shots.drop(columns=["FixedAngle"], errors='ignore'),
        on="ScoringWagon",
        how="left"
    ).fillna(0) 
    
    wagon_summary["ScoringWagon"] = pd.Categorical(wagon_summary["ScoringWagon"], categories=all_areas, ordered=True)
    wagon_summary = wagon_summary.sort_values("ScoringWagon").reset_index(drop=True)
    
    total_runs = wagon_summary["TotalRuns"].sum()
    if total_runs > 0:
        wagon_summary["RunPercentage"] = (wagon_summary["TotalRuns"] / total_runs) * 100
    else:
        wagon_summary["RunPercentage"] = 0 
        
    wagon_summary["FixedAngle"] = wagon_summary["FixedAngle"].astype(int) 

except KeyError as e:
    wagon_summary = pd.DataFrame()

# --- INTERCEPTION DATA FILTERING (Retained) ---
df_interception = filtered_df[filtered_df["InterceptionX"] > -999].copy()
df_interception["ColorType"] = "Other"
df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
color_map = {
    "Wicket": "red",
    "Boundary": "royalblue",
    "Other": "white" 
}

# ==============================================================================
# 5. CHART GRID LAYOUT (2 ROWS, 3 COLUMNS)
# ==============================================================================
# --- ROW 1: Zonal, Crease Beehive, Pitch Map ---
st.markdown("#### **Row 1:** Zonal, Crease Beehive, Pitch Map")
col_r1_c1, col_r1_c2, col_r1_c3 = st.columns(3)

# ------------------------------------------------------------------------------
# 1. CHART: ZONAL BOXES (CREASE BEEHIVE BOXES) - Col 1
# ------------------------------------------------------------------------------
with col_r1_c1:
    st.caption("**1. Zonal Analysis**")
    if filtered_df.empty:
        st.warning("No data.")
    else:
        # --- Zonal Calculation Logic (retained) ---
        right_hand_zones = {
            "Z1": (-0.72, 0, -0.45, 1.91), "Z2": (-0.45, 0, -0.18, 0.71),
            "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (-0.45, 0.71, -0.18, 1.31),
            "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.45, 1.31, 0.18, 1.91),
        }
        left_hand_zones = {
            "Z1": (0.45, 0, 0.72, 1.91), "Z2": (0.18, 0, 0.45, 0.71),
            "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (0.18, 0.71, 0.45, 1.31),
            "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.18, 1.31, 0.45, 1.91), 
        }

        handed_data = filtered_df["IsBatsmanRightHanded"].dropna().mode()
        is_right_handed = handed_data.iloc[0] if not handed_data.empty and batsman != "All" else True
        zones_layout = right_hand_zones if is_right_handed else left_hand_zones
        
        def assign_zone(row):
            x, y = row["CreaseY"], row["CreaseZ"]
            for zone, (x1, y1, x2, y2) in zones_layout.items():
                if x1 <= x <= x2 and y1 <= y <= y2: return zone
            return "Other"

        df_chart2 = filtered_df.copy()
        df_chart2["Zone"] = df_chart2.apply(assign_zone, axis=1)
        df_chart2 = df_chart2[df_chart2["Zone"] != "Other"]
        
        summary = (
            df_chart2.groupby("Zone")
            .agg(Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: (x == True).sum()))
            .reindex(["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).fillna(0)
        )
        summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)

        # --- Matplotlib Plotting: EXTREME REDUCTION ---
        avg_values = summary["Avg Runs/Wicket"]
        norm = mcolors.Normalize(vmin=avg_values[avg_values > 0].min(), vmax=avg_values.max()) if avg_values.max() > 0 else mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('Blues')
        
        # *** REDUCED SIZE ***
        fig_boxes, ax = plt.subplots(figsize=(2.5, 3.0)) # Width 2.5, Height 3.0

        for zone, (x1, y1, x2, y2) in zones_layout.items():
            w, h = x2 - x1, y2 - y1
            if zone not in summary.index: runs, wkts, avg, color = 0, 0, 0, 'white' 
            else:
                runs = int(summary.loc[zone, "Runs"]); wkts = int(summary.loc[zone, "Wickets"])
                avg = summary.loc[zone, "Avg Runs/Wicket"]; color = cmap(norm(avg))

            ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=0.5))

            ax.text(x1 + w / 2, y1 + h / 2,
                f"{zone}\nR:{runs} W:{wkts}\nA:{avg:.1f}",
                ha="center", va="center", weight="bold", 
                fontsize=5, # Reduced font size 
                color="black" if norm(avg) < 0.6 else "white"
            )
        
        ax.set_xlim(-0.75, 0.75); ax.set_ylim(0, 2); ax.axis('off') 
        plt.tight_layout(pad=0) 
        st.pyplot(fig_boxes)

# ------------------------------------------------------------------------------
# 2. CHART: CREASE BEEHIVE (SCATTER PLOT) - Col 2
# ------------------------------------------------------------------------------
with col_r1_c2:
    st.caption("**2. Crease Beehive**")
    if filtered_df.empty:
        st.warning("No data.")
    else:
        wickets = filtered_df[filtered_df["Wicket"] == True]; non_wickets = filtered_df[filtered_df["Wicket"] == False]
        fig_cbh = go.Figure()

        # Reduced marker size
        fig_cbh.add_trace(go.Scatter(x=non_wickets["StumpsY"], y=non_wickets["StumpsZ"], mode='markers', name="No Wicket",
            marker=dict(color='lightgrey', size=4, line=dict(width=0), opacity=0.8)))

        fig_cbh.add_trace(go.Scatter(x=wickets["StumpsY"], y=wickets["StumpsZ"], mode='markers', name="Wicket",
            marker=dict(color='red', size=6, line=dict(width=0), opacity=0.95)))

        # Keep only lines
        fig_cbh.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1))
        fig_cbh.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1))
        fig_cbh.add_vline(x=-0.92, line=dict(color="grey", width=0.8))
        fig_cbh.add_vline(x=0.92, line=dict(color="grey", width=0.8))
        fig_cbh.add_hline(y=0.78, line=dict(color="grey", width=0.8))
        fig_cbh.add_hline(y=1.31, line=dict(color="grey", width=0.8)) 

        # --- ADJUSTED SIZE (Fixed width, minimal height/margins) ---
        fig_cbh.update_layout(
            width=300, height=300, # EXTREME REDUCTION
            xaxis=dict(range=[-1.6, 1.6], showgrid=True, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[0.5, 2], showgrid=True, zeroline=False, visible=False),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=5, r=5, t=5, b=5), showlegend=False
        )
        st.plotly_chart(fig_cbh, use_container_width=False) 

# ------------------------------------------------------------------------------
# 3. CHART: PITCH MAP - Col 3
# ------------------------------------------------------------------------------
with col_r1_c3:
    st.caption("**3. Pitch Map (Bounce)**")
    if filtered_df.empty:
        st.warning("No data.")
    else:
        fig_pitch = go.Figure()
        PITCH_Y_LINES = [8.60, 5.0, 2.8, 0.9]; PITCH_Y_LABELS = ["Short", "Length", "Slot", "Yorker"]
        
        for y_val in PITCH_Y_LINES:
             fig_pitch.add_hline(y=y_val, line=dict(color="lightgrey", width=1.0, dash="dot"))
        
        for i, y_val in enumerate(PITCH_Y_LINES):
             fig_pitch.add_annotation(
                x=-1.4, y=y_val - 0.5 if i==0 else y_val - 1.8,
                text=PITCH_Y_LABELS[i], showarrow=False,
                font=dict(size=7, color="grey", weight='bold'), xanchor='left'
            )
             
        fig_pitch.add_hline(y=0.0, line=dict(color="black", width=2))
        fig_pitch.add_vline(x=-0.18, line=dict(color="#777777", dash="dot", width=1.2))
        fig_pitch.add_vline(x=0.18, line=dict(color="#777777", dash="dot", width=1.2))
        
        pitch_wickets = filtered_df[filtered_df["Wicket"] == True]; pitch_non_wickets = filtered_df[filtered_df["Wicket"] == False]

        fig_pitch.add_trace(go.Scatter(x=pitch_non_wickets["BounceY"], y=pitch_non_wickets["BounceX"], mode='markers', name="No Wicket",
            marker=dict(color='white', size=4, line=dict(width=0.5, color="grey"), opacity=0.8)))

        fig_pitch.add_trace(go.Scatter(x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"], mode='markers', name="Wicket",
            marker=dict(color='red', size=7, line=dict(width=0), opacity=0.95)))

        # --- ADJUSTED SIZE (Fixed width, minimal height/margins) ---
        fig_pitch.update_layout(
            width=300, height=300, # EXTREME REDUCTION
            xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False,visible = False),
            yaxis=dict(range=[16.0, -4.0], showgrid=False, zeroline=False,visible = False), 
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=5, r=5, t=5, b=5), showlegend=True,
            legend=dict(font=dict(size=7)) 
        )
        st.plotly_chart(fig_pitch, use_container_width=False) 

st.markdown("---")

# --- ROW 2: Interception Side, Interception Front, Wagon Wheel ---
st.markdown("#### **Row 2:** Interception Side, Interception Front, Wagon Wheel")
col_r2_c1, col_r2_c2, col_r2_c3 = st.columns(3)

# ------------------------------------------------------------------------------
# 4. CHART: INTERCEPTION POINTS (SIDE-ON - Vertical View) - Col 1
# ------------------------------------------------------------------------------
with col_r2_c1:
    st.caption("**4. Interception Side-On (H/D)**")
    if df_interception.empty:
        st.warning("No valid data.")
    else:
        # *** ADJUSTED SIZE ***
        fig_7, ax_7 = plt.subplots(figsize=(3, 3)) # Square plot for balance
        
        df_other = df_interception[df_interception["ColorType"] == "Other"]
        ax_7.scatter(df_other["InterceptionX"] + 10, df_other["InterceptionZ"], 
                     color='white', edgecolors='grey', linewidths=0.5, s=15, label="Other") 
        
        for ctype in ["Boundary", "Wicket"]:
            df_slice = df_interception[df_interception["ColorType"] == ctype]
            ax_7.scatter(df_slice["InterceptionX"] + 10, df_slice["InterceptionZ"], 
                         color=color_map[ctype], s=25, label=ctype) 

        # Draw Dashed Lines
        line_specs = {0.0: "Stumps", 1.250: "Crease", 2.000: "2m", 3.000: "3m"}
        for x_val, label in line_specs.items():
            ax_7.axvline(x=x_val, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)    
            ax_7.text(x_val, 1.45, label.split(':')[-1].strip(), 
                      ha='center', va='center', fontsize=5, color='grey', 
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        ax_7.set_xlim(-0.2, 3.4); ax_7.set_ylim(0, 1.5) 
        ax_7.tick_params(axis='y', which='both', labelleft=False, left=False)
        ax_7.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
        ax_7.set_xlabel("Distance (m)", fontsize=7) 
        ax_7.set_ylabel("Height (m)", fontsize=7) 
        ax_7.legend(loc='upper right', fontsize=6) 
        ax_7.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout(pad=0)
        st.pyplot(fig_7)

# ------------------------------------------------------------------------------
# 5. CHART: INTERCEPTION POINTS (TOP-DOWN) - Col 2
# ------------------------------------------------------------------------------
with col_r2_c2:
    st.caption("**5. Interception Front-On (W/D)**")
    if df_interception.empty:
        st.warning("No valid data.")
    else:
        # *** ADJUSTED SIZE ***
        fig_8, ax_8 = plt.subplots(figsize=(3, 3)) # Square plot for balance
        
        df_other = df_interception[df_interception["ColorType"] == "Other"]
        ax_8.scatter(df_other["InterceptionY"], df_other["InterceptionX"] + 10, 
                     color='white', edgecolors='grey', linewidths=0.5, s=15, label="Other") 
        
        for ctype in ["Boundary", "Wicket"]:
            df_slice = df_interception[df_interception["ColorType"] == ctype]
            ax_8.scatter(df_slice["InterceptionY"], df_slice["InterceptionX"] + 10, 
                         color=color_map[ctype], s=25, label=ctype) 

        # Draw Horizontal Dashed Lines
        line_specs = {0.00: "Stumps", 1.25: "Crease"}
        for y_val, label in line_specs.items():
            ax_8.axhline(y=y_val, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
            ax_8.text(-0.95, y_val, label.split(':')[-1].strip(), 
                      ha='left', va='center', fontsize=6, color='grey', 
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # Draw Vertical Solid Stumps Lines
        ax_8.axvline(x=-0.18, color='grey', linestyle='-', linewidth=1.5, alpha=0.7)
        ax_8.axvline(x= 0.18, color='grey', linestyle='-', linewidth=1.5, alpha=0.7)
        
        ax_8.set_xlim(-1, 1); ax_8.set_ylim(-0.2, 3.5); ax_8.invert_yaxis()      
        
        ax_8.tick_params(axis='y', which='both', labelleft=False, left=False)
        ax_8.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
        ax_8.set_xlabel("Width (m)", fontsize=7) 
        ax_8.set_ylabel("Distance (m)", fontsize=7) 
        ax_8.legend(loc='lower right', fontsize=6) 
        ax_8.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout(pad=0)
        st.pyplot(fig_8)

# ------------------------------------------------------------------------------
# 6. CHART: SCORING AREAS (WAGON WHEEL) - Col 3
# ------------------------------------------------------------------------------
with col_r2_c3:
    st.caption("**6. Scoring Areas (Wagon)**")
    if wagon_summary.empty:
        st.warning("No scoring shots.")
    else:
        angles = wagon_summary["FixedAngle"].tolist()
        runs = wagon_summary["TotalRuns"].tolist()
        labels = [f"{area}\n({pct:.0f}%)" for area, pct in zip(wagon_summary["ScoringWagon"], wagon_summary["RunPercentage"])]
        
        run_min = min(runs); run_max = max(runs)
        norm = mcolors.Normalize(vmin=run_min, vmax=run_max) if run_max > run_min else mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('Greens')
        colors = cmap(norm(runs))
        
        for i, run_count in enumerate(runs):
            if run_count == 0:
                colors[i] = (1.0, 1.0, 1.0, 1.0) # White for 0 runs

        # *** ADJUSTED SIZE ***
        fig, ax = plt.subplots(figsize=(3, 3)) # Square plot for balance
        
        # Create the Pie Chart
        wedges, texts = ax.pie(
            angles, 
            colors=colors, 
            wedgeprops={"width": 1, "edgecolor": "black", "linewidth": 0.5},
            startangle=90, 
            counterclock=False, 
            labels=labels, 
            labeldistance=1.1, 
        )
        
        # Customize Text Properties
        for text in texts:
            text.set_color('black')
            text.set_fontsize(7) # Significantly reduced font size
            text.set_fontweight('bold')

        ax.axis('equal') 
        plt.tight_layout(pad=0)
        st.pyplot(fig)

st.markdown("---")
st.caption("Dashboard End.")
