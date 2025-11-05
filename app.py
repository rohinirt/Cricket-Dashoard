import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# --- 1. SET PAGE CONFIGURATION ---
st.set_page_config(page_title="Cricket Analysis Dashboard (Comparative)", layout="wide")

# --- 2. INJECT CUSTOM CSS FOR ABSOLUTE MINIMUM SPACING ---
st.markdown("""
<style>
    /* ... (CSS for compactness, refined slightly) ... */
    .block-container {
        padding-top: 0.1rem; 
        padding-right: 0.1rem;
        padding-left: 0.1rem;
        padding-bottom: 0.1rem;
    }
    
    h1, h2, h3, h4, p, .stCaption {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    .st-emotion-cache-1om0885 { /* For st.columns internal padding */
        padding: 0px !important; 
    }

    .st-emotion-cache-n3wfc { /* For the main st.columns wrapper */
        padding-left: 0.2rem;
        padding-right: 0.2rem;
    }
    
    hr {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    
    /* Center the title, if needed */
    .st-emotion-cache-nahz7x { /* Adjust this selector if your title is not centered */
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- WAGON WHEEL UTILITY FUNCTIONS (Retained) ---
def calculate_scoring_wagon(row):
    LX = row.get("LandingX"); LY = row.get("LandingY"); RH = row.get("IsBatsmanRightHanded")
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0: return None
    def atan_safe(numerator, denominator): return np.arctan(numerator / denominator) if denominator != 0 else np.nan 
    
    if RH == True: # Right Handed Batsman
        if LX <= 0 and LY > 0: return "FINE LEG";
        elif LX <= 0 and LY <= 0: return "THIRD MAN";
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "COVER";
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG OFF";
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "SQUARE LEG";
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG ON";
    elif RH == False: # Left Handed Batsman
        if LX <= 0 and LY > 0: return "THIRD MAN";
        elif LX <= 0 and LY <= 0: return "FINE LEG";
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "SQUARE LEG";
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG ON";
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "COVER";
            elif atan_safe(LX, LY) <= np.pi / 4: return "LONG OFF";
    return None

def calculate_scoring_angle(area):
    if area in ["FINE LEG", "THIRD MAN"]: return 90
    elif area in ["COVER", "SQUARE LEG", "LONG OFF", "LONG ON"]: return 45
    return 0 

# --- CHART GENERATION FUNCTION (REFACTORED FOR NEW LAYOUT) ---
def generate_charts(df_input, panel_title, panel_col, batsman_name):
    """Generates the set of charts within the specified Streamlit column based on the reference layout."""
    
    # 1. DATA PREP
    # Ensure a copy to avoid SettingWithCopyWarning
    df = df_input.copy() 

    # Filter for Interception data
    df_interception = df[df["InterceptionX"] > -999].copy()
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}

    # Wagon Wheel Data Prep
    wagon_summary = pd.DataFrame() 
    try:
        df_wagon = df.copy()
        df_wagon["ScoringWagon"] = df_wagon.apply(calculate_scoring_wagon, axis=1)
        df_wagon["FixedAngle"] = df_wagon["ScoringWagon"].apply(calculate_scoring_angle)
        
        summary_with_shots = df_wagon.groupby("ScoringWagon").agg(
            TotalRuns=("Runs", "sum"), FixedAngle=("FixedAngle", 'first')
        ).reset_index().dropna(subset=["ScoringWagon"])
        
        handedness_mode = df["IsBatsmanRightHanded"].dropna().mode()
        is_right_handed = handedness_mode.iloc[0] if not handedness_mode.empty and batsman_name != "All" else True
        
        all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"] if is_right_handed else ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]
        template_df = pd.DataFrame({"ScoringWagon": all_areas, "FixedAngle": [calculate_scoring_angle(area) for area in all_areas]})

        wagon_summary = template_df.merge(summary_with_shots.drop(columns=["FixedAngle"], errors='ignore'), on="ScoringWagon", how="left").fillna(0) 
        wagon_summary["ScoringWagon"] = pd.Categorical(wagon_summary["ScoringWagon"], categories=all_areas, ordered=True)
        wagon_summary = wagon_summary.sort_values("ScoringWagon").reset_index(drop=True)
        
        total_runs = wagon_summary["TotalRuns"].sum()
        wagon_summary["RunPercentage"] = (wagon_summary["TotalRuns"] / total_runs) * 100 if total_runs > 0 else 0 
        wagon_summary["FixedAngle"] = wagon_summary["FixedAngle"].astype(int) 
    except Exception:
        pass 

    # Zonal Analysis Prep
    right_hand_zones = {"Z1": (-0.72, 0, -0.45, 1.91), "Z2": (-0.45, 0, -0.18, 0.71), "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (-0.45, 0.71, -0.18, 1.31), "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.45, 1.31, 0.18, 1.91)}
    left_hand_zones = {"Z1": (0.45, 0, 0.72, 1.91), "Z2": (0.18, 0, 0.45, 0.71), "Z3": (-0.18, 0, 0.18, 0.71), "Z4": (0.18, 0.71, 0.45, 1.31), "Z5": (-0.18, 0.71, 0.18, 1.31), "Z6": (-0.18, 1.31, 0.45, 1.91)}
    zones_layout = right_hand_zones if is_right_handed else left_hand_zones
    
    def assign_zone(row):
        x, y = row["CreaseY"], row["CreaseZ"]
        for zone, (x1, y1, x2, y2) in zones_layout.items():
            if x1 <= x <= x2 and y1 <= y <= y2: return zone
        return "Other"

    df_chart2 = df.copy(); df_chart2["Zone"] = df_chart2.apply(assign_zone, axis=1)
    summary_zonal = df_chart2[df_chart2["Zone"] != "Other"].groupby("Zone").agg(Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: (x == True).sum())).reindex(["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]).fillna(0)
    summary_zonal["Avg Runs/Wicket"] = summary_zonal.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)

    # Calculate run percentages for "Hit Zone" style chart
    runs_total = df["Runs"].sum()
    runs_boundary = df[df["Runs"].isin([4, 6])]["Runs"].sum()
    runs_single_double = df[df["Runs"].isin([1, 2, 3])]["Runs"].sum()
    runs_dot = df[df["Runs"] == 0].shape[0] - df[df["Wicket"] == True].shape[0] # Dot balls not resulting in wicket

    # Assuming the "out side off/leg side" for the 'hit zone' chart refers to the InterceptionY
    # We'll categorize based on InterceptionY relative to the stumps
    # This is a simplified categorization, you might need to refine based on your data's definition
    
    off_side_balls = df[df["InterceptionY"] < -0.18].shape[0] # To the left of off stump
    leg_side_balls = df[df["InterceptionY"] > 0.18].shape[0] # To the right of leg stump
    straight_balls = df[(df["InterceptionY"] >= -0.18) & (df["InterceptionY"] <= 0.18)].shape[0] # Between stumps

    total_interceptions = off_side_balls + leg_side_balls + straight_balls
    off_side_pct = (off_side_balls / total_interceptions * 100) if total_interceptions > 0 else 0
    leg_side_pct = (leg_side_balls / total_interceptions * 100) if total_interceptions > 0 else 0
    straight_pct = (straight_balls / total_interceptions * 100) if total_interceptions > 0 else 0

    # Summary Stats for the bottom bar charts
    avg_speed = df["LandingX"].mean() # Placeholder, use actual speed column if available
    avg_bounce_height = df["BounceY"].mean() # Placeholder
    avg_wickets_per_over = df.groupby("Over")["Wicket"].sum().mean() # Placeholder

    # --- Start Chart Rendering ---
    panel_col.markdown(f"**{panel_title}**")
    # panel_col.markdown("---") # Removed as per reference image

    # ROW 1: PITCH MAP (Wider Chart)
    # Using a single column within the panel, or a column with specific width
    with panel_col: # This ensures it uses the full width of the main panel column
        # st.caption("**Pitch Map (Bounce Location)**") # Removed for compactness
        if df.empty:
            st.warning("No data for Pitch Map.")
        else:
            fig_pitch = go.Figure()
            PITCH_Y_LINES = [8.60, 5.0, 2.8, 0.9]; PITCH_Y_LABELS = ["Short", "Length", "Slot", "Yorker"]
            
            for y_val in PITCH_Y_LINES: fig_pitch.add_hline(y=y_val, line=dict(color="lightgrey", width=1.0, dash="dot"))
            for i, y_val in enumerate(PITCH_Y_LINES):
                fig_pitch.add_annotation(x=-1.4, y=y_val - 0.5 if i==0 else y_val - 1.8, text=PITCH_Y_LABELS[i], showarrow=False,
                    font=dict(size=7, color="grey", weight='bold'), xanchor='left')
                 
            fig_pitch.add_hline(y=0.0, line=dict(color="black", width=2))
            fig_pitch.add_vline(x=-0.18, line=dict(color="#777777", dash="dot", width=1.2)); fig_pitch.add_vline(x=0.18, line=dict(color="#777777", dash="dot", width=1.2))
            
            pitch_wickets = df[df["Wicket"] == True]; pitch_non_wickets = df[df["Wicket"] == False]
            fig_pitch.add_trace(go.Scatter(x=pitch_non_wickets["BounceY"], y=pitch_non_wickets["BounceX"], mode='markers', name="No Wicket", marker=dict(color='white', size=4, line=dict(width=0.5, color="grey"), opacity=0.8)))
            fig_pitch.add_trace(go.Scatter(x=pitch_wickets["BounceY"], y=pitch_wickets["BounceX"], mode='markers', name="Wicket", marker=dict(color='red', size=7, line=dict(width=0), opacity=0.95)))

            # ADJUSTED SIZE - Wider, but still compact
            fig_pitch.update_layout(width=320, height=250, # Adjusted dimensions for wider look
                xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False,visible = False),
                yaxis=dict(range=[16.0, -4.0], showgrid=False, zeroline=False,visible = False), 
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=5, r=5, t=5, b=5), showlegend=True,
                legend=dict(font=dict(size=7), x=1.0, y=1.0, xanchor='right', yanchor='top') # Legend at top-right
            )
            st.plotly_chart(fig_pitch, use_container_width=False) 

    # ROW 2: CREASE BEEHIVE & ZONAL ANALYSIS (Two side-by-side)
    col_r2_c1, col_r2_c2 = panel_col.columns([0.6, 0.4]) # Adjust width ratio
    with col_r2_c1:
        # st.caption("**Crease Beehive**") # Removed for compactness
        if df.empty: st.warning("No data.")
        else:
            wickets = df[df["Wicket"] == True]; non_wickets = df[df["Wicket"] == False]
            fig_cbh = go.Figure()
            fig_cbh.add_trace(go.Scatter(x=non_wickets["StumpsY"], y=non_wickets["StumpsZ"], mode='markers', name="No Wicket", marker=dict(color='lightgrey', size=4, line=dict(width=0), opacity=0.8)))
            fig_cbh.add_trace(go.Scatter(x=wickets["StumpsY"], y=wickets["StumpsZ"], mode='markers', name="Wicket", marker=dict(color='red', size=6, line=dict(width=0), opacity=0.95)))

            fig_cbh.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1)); fig_cbh.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1))
            fig_cbh.add_vline(x=-0.92, line=dict(color="grey", width=0.8)); fig_cbh.add_vline(x=0.92, line=dict(color="grey", width=0.8))
            fig_cbh.add_hline(y=0.78, line=dict(color="grey", width=0.8)); fig_cbh.add_hline(y=1.31, line=dict(color="grey", width=0.8)) 

            fig_cbh.update_layout(width=200, height=200, # Smaller dimensions
                xaxis=dict(range=[-1.6, 1.6], showgrid=True, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[0.5, 2], showgrid=True, zeroline=False, visible=False),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=5, r=5, t=5, b=5), showlegend=False
            )
            st.plotly_chart(fig_cbh, use_container_width=False) 
    
    with col_r2_c2:
        # st.caption("**Zonal Analysis**") # Removed for compactness
        if df.empty: st.warning("No data.")
        else:
            avg_values = summary_zonal["Avg Runs/Wicket"]
            norm = mcolors.Normalize(vmin=avg_values[avg_values > 0].min(), vmax=avg_values.max()) if avg_values.max() > 0 else mcolors.Normalize(vmin=0, vmax=1)
            cmap = cm.get_cmap('Blues')
            
            fig_boxes, ax = plt.subplots(figsize=(2.0, 2.0)) # Smaller dimensions, square
            for zone, (x1, y1, x2, y2) in zones_layout.items():
                w, h = x2 - x1, y2 - y1
                runs, wkts, avg = (int(summary_zonal.loc[zone, "Runs"]), int(summary_zonal.loc[zone, "Wickets"]), summary_zonal.loc[zone, "Avg Runs/Wicket"]) if zone in summary_zonal.index else (0, 0, 0)
                color = cmap(norm(avg)) if avg > 0 else 'white'

                ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=0.5))
                ax.text(x1 + w / 2, y1 + h / 2, f"{zone}\nR:{runs} W:{wkts}\nA:{avg:.1f}", ha="center", va="center", weight="bold", fontsize=4, color="black" if norm(avg) < 0.6 else "white")
            
            ax.set_xlim(-0.75, 0.75); ax.set_ylim(0, 2); ax.axis('off'); plt.tight_layout(pad=0) 
            st.pyplot(fig_boxes)

    # ROW 3: INTERCEPTION FRONT-ON & SIDE-ON (Two side-by-side)
    col_r3_c1, col_r3_c2 = panel_col.columns([0.5, 0.5]) # Equal width
    with col_r3_c1:
        # st.caption("**Interception Front-On**") # Removed for compactness
        if df_interception.empty: st.warning("No valid data.")
        else:
            fig_8, ax_8 = plt.subplots(figsize=(2.5, 2.5)) # Square plot 
            df_other = df_interception[df_interception["ColorType"] == "Other"]
            ax_8.scatter(df_other["InterceptionY"], df_other["InterceptionX"] + 10, color='white', edgecolors='grey', linewidths=0.5, s=15, label="Other") 
            
            for ctype in ["Boundary", "Wicket"]:
                df_slice = df_interception[df_interception["ColorType"] == ctype]
                ax_8.scatter(df_slice["InterceptionY"], df_slice["InterceptionX"] + 10, color=color_map[ctype], s=25, label=ctype) 

            line_specs = {0.00: "Stumps", 1.25: "Crease"}
            for y_val, label in line_specs.items():
                ax_8.axhline(y=y_val, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
                ax_8.text(-0.95, y_val, label.split(':')[-1].strip(), ha='left', va='center', fontsize=6, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

            ax_8.axvline(x=-0.18, color='grey', linestyle='-', linewidth=1.5, alpha=0.7); ax_8.axvline(x= 0.18, color='grey', linestyle='-', linewidth=1.5, alpha=0.7)
            
            ax_8.set_xlim(-1, 1); ax_8.set_ylim(-0.2, 3.5); ax_8.invert_yaxis()      
            ax_8.tick_params(axis='y', which='both', labelleft=False, left=False); ax_8.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
            ax_8.set_xlabel("Width (m)", fontsize=7); ax_8.set_ylabel("Distance (m)", fontsize=7) 
            ax_8.legend(loc='lower right', fontsize=6); ax_8.grid(True, linestyle=':', alpha=0.5); plt.tight_layout(pad=0)
            st.pyplot(fig_8)

    with col_r3_c2:
        # st.caption("**Interception Side-On**") # Removed for compactness
        if df_interception.empty: st.warning("No valid data.")
        else:
            fig_7, ax_7 = plt.subplots(figsize=(2.5, 2.5)) # Square plot 
            df_other = df_interception[df_interception["ColorType"] == "Other"]
            ax_7.scatter(df_other["InterceptionX"] + 10, df_other["InterceptionZ"], color='white', edgecolors='grey', linewidths=0.5, s=15, label="Other") 
            
            for ctype in ["Boundary", "Wicket"]:
                df_slice = df_interception[df_interception["ColorType"] == ctype]
                ax_7.scatter(df_slice["InterceptionX"] + 10, df_slice["InterceptionZ"], color=color_map[ctype], s=25, label=ctype) 

            line_specs = {0.0: "Stumps", 1.250: "Crease", 2.000: "2m", 3.000: "3m"}
            for x_val, label in line_specs.items():
                ax_7.axvline(x=x_val, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)    
                ax_7.text(x_val, 1.45, label.split(':')[-1].strip(), ha='center', va='center', fontsize=5, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

            ax_7.set_xlim(-0.2, 3.4); ax_7.set_ylim(0, 1.5) 
            ax_7.tick_params(axis='y', which='both', labelleft=False, left=False); ax_7.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
            ax_7.set_xlabel("Distance (m)", fontsize=7); ax_7.set_ylabel("Height (m)", fontsize=7) 
            ax_7.legend(loc='upper right', fontsize=6); ax_7.grid(True, linestyle=':', alpha=0.5); plt.tight_layout(pad=0)
            st.pyplot(fig_7)

    # ROW 4: WAGON WHEEL & CUSTOM HIT ZONE BAR CHART
    col_r4_c1, col_r4_c2 = panel_col.columns([0.5, 0.5]) # Equal width
    with col_r4_c1:
        # st.caption("**Scoring Areas (Wagon)**") # Removed for compactness
        if wagon_summary.empty: st.warning("No scoring shots.")
        else:
            angles = wagon_summary["FixedAngle"].tolist(); runs = wagon_summary["TotalRuns"].tolist()
            labels = [f"{area}\n({pct:.0f}%)" for area, pct in zip(wagon_summary["ScoringWagon"], wagon_summary["RunPercentage"])]
            
            run_min = min(runs); run_max = max(runs)
            norm = mcolors.Normalize(vmin=run_min, vmax=run_max) if run_max > run_min else mcolors.Normalize(vmin=0, vmax=1)
            cmap = cm.get_cmap('Greens')
            colors = cmap(norm(runs))
            for i, run_count in enumerate(runs):
                if run_count == 0: colors[i] = (1.0, 1.0, 1.0, 1.0)

            fig, ax = plt.subplots(figsize=(2.5, 2.5)) # Square plot 
            wedges, texts = ax.pie(angles, colors=colors, wedgeprops={"width": 1, "edgecolor": "black", "linewidth": 0.5}, startangle=90, counterclock=False, labels=labels, labeldistance=1.1)
            
            for text in texts:
                text.set_color('black'); text.set_fontsize(7); text.set_fontweight('bold')

            ax.axis('equal'); plt.tight_layout(pad=0)
            st.pyplot(fig)
    
    with col_r4_c2:
        # Custom "Hit Zone" style chart (Example based on InterceptionY)
        # st.caption("**Ball Locations (%)**") # Removed for compactness
        fig_hit_zone, ax_hit_zone = plt.subplots(figsize=(2.5, 2.5)) # Square plot
        zones = ["Off Side", "Straight", "Leg Side"]
        percentages = [off_side_pct, straight_pct, leg_side_pct]
        
        # Adjust colors for clarity
        zone_colors = ['#FF6347', '#9ACD32', '#6495ED'] # Tomato, YellowGreen, CornflowerBlue

        ax_hit_zone.barh(zones, percentages, color=zone_colors, height=0.7)
        ax_hit_zone.set_xlim(0, 100)
        ax_hit_zone.set_xticks([]) # Hide x-axis ticks
        ax_hit_zone.tick_params(axis='y', length=0, labelsize=7) # Smaller labels
        ax_hit_zone.invert_yaxis() # Top-most category first
        ax_hit_zone.spines['top'].set_visible(False)
        ax_hit_zone.spines['right'].set_visible(False)
        ax_hit_zone.spines['bottom'].set_visible(False)
        ax_hit_zone.spines['left'].set_visible(False)

        for i, (zone, pct) in enumerate(zip(zones, percentages)):
            ax_hit_zone.text(pct + 2, i, f'{pct:.0f}%', va='center', fontsize=7, color='black', fontweight='bold')
        
        plt.tight_layout(pad=0)
        st.pyplot(fig_hit_zone)

    # ROW 5: HORIZONTAL BAR CHARTS (3 side-by-side or stacked, adjusting for compactness)
    # The reference image has 3 charts, each with a single bar. We will simulate this.
    # We will use st.columns(3) for this as well, each holding a bar chart
    col_r5_c1, col_r5_c2, col_r5_c3 = panel_col.columns(3)

    # Bar chart 1: Average Speed
    with col_r5_c1:
        # st.caption("**Avg Speed**") # Removed for compactness
        fig_speed, ax_speed = plt.subplots(figsize=(1.5, 0.5)) # Very compact
        ax_speed.barh([''], [avg_speed], color='lightgray', height=0.5)
        ax_speed.set_xlim(0, max(100, avg_speed + 10)) # Adjust max x-limit
        ax_speed.text(0.5, 0, f'{avg_speed:.1f}', va='center', ha='left', fontsize=6, color='black', fontweight='bold')
        ax_speed.axis('off') # Hide axes
        plt.tight_layout(pad=0)
        st.pyplot(fig_speed)

    # Bar chart 2: Average Bounce Height
    with col_r5_c2:
        # st.caption("**Avg Bounce Height**") # Removed for compactness
        fig_bounce, ax_bounce = plt.subplots(figsize=(1.5, 0.5))
        ax_bounce.barh([''], [avg_bounce_height], color='lightgray', height=0.5)
        ax_bounce.set_xlim(0, max(2, avg_bounce_height + 0.5))
        ax_bounce.text(0.05, 0, f'{avg_bounce_height:.1f}', va='center', ha='left', fontsize=6, color='black', fontweight='bold')
        ax_bounce.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig_bounce)

    # Bar chart 3: Avg Wickets per Over
    with col_r5_c3:
        # st.caption("**Avg Wickets/Over**") # Removed for compactness
        fig_wickets, ax_wickets = plt.subplots(figsize=(1.5, 0.5))
        ax_wickets.barh([''], [avg_wickets_per_over], color='lightgray', height=0.5)
        ax_wickets.set_xlim(0, max(1, avg_wickets_per_over + 0.5))
        ax_wickets.text(0.05, 0, f'{avg_wickets_per_over:.1f}', va='center', ha='left', fontsize=6, color='black', fontweight='bold')
        ax_wickets.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig_wickets)

# --- GLOBAL DATA UPLOADER AND INITIAL LOAD (CRITICAL BLOCK) ---
uploaded_file = st.file_uploader("Upload your data", type=["csv"])

df = None 

if uploaded_file is None:
    st.info("👆 Please upload a **CSV file** with the required Hawkeye data columns to view the dashboard.")
    st.stop()

# --- DATA PROCESSING AFTER FILE IS UPLOADED ---
try:
    df = pd.read_csv(uploaded_file)
    required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", "InterceptionZ", "InterceptionY", "Over"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing one or more required columns. Required: {', '.join(required_cols)}. Please ensure your columns are named correctly.")
        st.stop()
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop() 

# --- TOP FILTERS (COMPACT ROW) ---
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])

with filter_col1:
    bat_team_options = ["All"] + sorted(df["BattingTeam"].dropna().unique().tolist())
    bat_team = st.selectbox("Select Batting Team", bat_team_options)
    df_bat_team = df if bat_team == "All" else df[df["BattingTeam"] == bat_team]

with filter_col2:
    batsman_options = ["All"] + sorted(df_bat_team["BatsmanName"].dropna().unique().tolist())
    batsman = st.selectbox("Select Batsman", batsman_options)
    df_batsman = df_bat_team if batsman == "All" else df_bat_team[df_bat_team["BatsmanName"] == batsman]

with filter_col3:
    over_options = ["All"] + sorted(df_batsman["Over"].dropna().unique().tolist())
    selected_over = st.selectbox("Select Over", over_options)
    df_over = df_batsman if selected_over == "All" else df_batsman[df_batsman["Over"] == selected_over]

st.markdown(f"<h2 style='text-align: center;'>{batsman.upper()}</h2>", unsafe_allow_html=True)
st.markdown("---")

# --- MAIN COMPARATIVE LAYOUT ---
col_seam, col_spin = st.columns(2)

# --- LEFT PANEL: SEAM ANALYSIS ---
filtered_df_seam = df_over[df_over["DeliveryType"] == "Seam"]
with col_seam:
    generate_charts(filtered_df_seam, "OUTSIDE OFF LEG SIDE", col_seam, batsman) # Panel title as per reference

# --- RIGHT PANEL: SPIN ANALYSIS ---
filtered_df_spin = df_over[df_over["DeliveryType"] == "Spin"]
with col_spin:
    generate_charts(filtered_df_spin, "VERY FINE OFF LONG OFF", col_spin, batsman) # Panel title as per reference

st.markdown("---")
st.caption("Dashboard End. Data is filtered by Team, Batsman, and Over (if selected) for both panels.")
