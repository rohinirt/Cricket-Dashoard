import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
# --- NEW IMPORTS FOR PDF GENERATION ---
from fpdf import FPDF 
from io import StringIO, BytesIO
import plotly.io as pio 

# --- 1. GLOBAL UTILITY FUNCTIONS ---

# Required columns check
REQUIRED_COLS = [
    "BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", 
    "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", 
    "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", 
    "InterceptionZ", "InterceptionY", "Over", 
    # **UPDATED COLUMNS REQUIRED FOR FILTERS AND DIRECTIONAL CHARTS**
    "Innings", "IsBowlerRightHanded", "Swing", "Deviation" 
]

# Function to encode Matplotlib figure to image for Streamlit (kept empty, no longer strictly needed)
def fig_to_image(fig):
    return fig

# --- PDF HELPERS ---

def fig_to_png_bytes(fig, is_plotly=False):
    """Converts a figure (Matplotlib or Plotly) to PNG bytes."""
    buf = BytesIO()
    if is_plotly:
        # Requires 'kaleido' library (pip install kaleido)
        pio.write_image(fig, buf, format='png', width=600, height=450) 
    else:
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

# --- CHART 1: ZONAL ANALYSIS (CBH Boxes) ---
# ... (Function definition for create_zonal_analysis remains here)

# --- CHART 2a: CREASE BEEHIVE ---
# ... (Function definition for create_crease_beehive remains here)

# --- CHART 2b: LATERAL PERFORMANCE STACKED BAR ---
def create_lateral_performance_boxes(df_in, delivery_type, batsman_name):
    # This function definition is included here to ensure completeness
    from matplotlib import cm, colors, patches
    
    df_lateral = df_in.copy()
    if df_lateral.empty:
        fig, ax = plt.subplots(figsize=(7, 1)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig     
    # 1. Define Zoning Logic (Same as before)
    def assign_lateral_zone(row):
        y = row["CreaseY"]
        if row["IsBatsmanRightHanded"] == True:
            if y > 0.18: return "LEG"
            elif y >= -0.18: return "STUMPS"
            elif y > -0.65: return "OUTSIDE OFF"
            else: return "WAY OUTSIDE OFF"
        else: # Left-Handed
            if y > 0.65: return "WAY OUTSIDE OFF"
            elif y > 0.18: return "OUTSIDE OFF"
            elif y >= -0.18: return "STUMPS"
            else: return "LEG"
    
    df_lateral["LateralZone"] = df_lateral.apply(assign_lateral_zone, axis=1)
    
    # 2. Calculate Summary Metrics
    summary = (
        df_lateral.groupby("LateralZone").agg(
            Runs=("Runs", "sum"), 
            Wickets=("Wicket", lambda x: (x == True).sum()), 
            Balls=("Wicket", "count")
        )
    )
    # ... (Rest of the function logic remains the same, calculates AVG, plots boxes)
    ordered_zones = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
    summary = summary.reindex(ordered_zones).fillna(0)

    # Calculate Average for coloring and labeling
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1)
    
    # 3. Chart Setup
    fig_boxes, ax_boxes = plt.subplots(figsize=(7, 1)) 
    
    num_regions = len(ordered_zones)
    box_width = 1 / num_regions 
    left = 0
    
    # Color Normalization (based on Average)
    avg_values = summary["Avg Runs/Wicket"]
    avg_max = avg_values.max() if avg_values.max() > 0 else 50
    norm = mcolors.Normalize(vmin=0, vmax=avg_max if avg_max > 50 else 50)
    cmap = cm.get_cmap('Reds') 
    
    # 4. Plotting Equal Boxes (Horizontal Heatmap)
    for index, row in summary.iterrows():
        avg = row["Avg Runs/Wicket"]
        wkts = int(row["Wickets"])
        
        color = cmap(norm(avg)) if row["Balls"] > 0 else 'whitesmoke' 
        
        ax_boxes.add_patch(
            patches.Rectangle((left, 0), box_width, 1, 
                              edgecolor="black", facecolor=color, linewidth=1)
        )
        
        label_wkts_avg = f"{wkts}W - Ave {avg:.1f}"
        
        if row["Balls"] > 0:
            r, g, b, a = color 
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
        else:
            text_color = 'black' 

        ax_boxes.text(left + box_width / 2, 0.75, 
                      index,
                      ha='center', va='center', fontsize=10, color=text_color)
                      
        ax_boxes.text(left + box_width / 2, 0.4, 
                      label_wkts_avg,
                      ha='center', va='center', fontsize= 10, fontweight = 'bold', color=text_color)
        
        left += box_width
        
    ax_boxes.set_xlim(0, 1); ax_boxes.set_ylim(0, 1)
    ax_boxes.axis('off') 
    ax_boxes.spines['right'].set_visible(False)
    ax_boxes.spines['top'].set_visible(False)
    ax_boxes.spines['left'].set_visible(False)
    ax_boxes.spines['bottom'].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    return fig_boxes
    
# --- CHART 3: PITCH MAP ---
# ... (Function definition for get_pitch_bins, create_pitch_map, create_pitch_length_run_pct remains here)

# --- CHART 4a: INTERCEPTION SIDE-ON --- 
# ... (Function definition for create_interception_side_on remains here)

# --- Chart 4b: Interception Side on Bins ---
# ... (Function definition for create_crease_width_split remains here)

# --- CHART 5: INTERCEPTION FRONT-ON --- 
# ... (Function definition for create_interception_front_on remains here)

# --- CHART 6: SCORING WAGON WHEEL ---
# ... (Function definition for calculate_scoring_wagon, create_wagon_wheel remains here)

# --- CHART 7: LEFT/RIGHT SCORING SPLIT (100% Bar) ---
# ... (Function definition for create_left_right_split remains here)

# --- CHART 9/10: DIRECTIONAL SPLIT (Side-by-Side Bars) ---
# ... (Function definition for create_directional_split remains here)


# =========================================================
# **PDF COMPILER FUNCTION (THE MISSING LOGIC)**
# =========================================================

def create_pdf_report(figures, batsman_name):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Title Page/Header
    pdf.cell(0, 10, f"Performance Report: {batsman_name.upper()}", 0, 1, 'C')
    pdf.ln(5)

    # --- SEAM ANALYSIS ---
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(213, 34, 33) # Deep Red
    pdf.cell(0, 10, "SEAM ANALYSIS", 0, 1, 'L')
    pdf.set_text_color(0, 0, 0) # Reset color
    pdf.ln(2)

    # Row 1: Zonal (100mm wide) + Pitch Run % (50mm wide)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(100, 5, "1. CREASE BEEHIVE ZONES (SR)", 0, 0, 'L')
    pdf.cell(50, 5, "2. PITCH LENGTH RUN %", 0, 1, 'L')
    y_start = pdf.get_y()
    pdf.image(fig_to_png_bytes(figures['seam_zonal']), x=10, y=y_start, w=100, type='PNG')
    pdf.image(fig_to_png_bytes(figures['seam_pitch_pct']), x=115, y=y_start, w=50, type='PNG')
    pdf.set_y(y_start + 75)
    
    # Row 2: Beehive (100mm wide) + Lateral Boxes (85mm wide)
    pdf.cell(0, 5, "3. CREASE BEEHIVE", 0, 0, 'L')
    pdf.set_x(105)
    pdf.cell(0, 5, "4. LATERAL PERFORMANCE BOXES", 0, 1, 'L')
    y_start = pdf.get_y()
    pdf.image(fig_to_png_bytes(figures['seam_beehive'], is_plotly=True), x=10, y=y_start, w=100, type='PNG')
    pdf.image(fig_to_png_bytes(figures['seam_lateral']), x=105, y=y_start + 5, w=85, type='PNG') # Vertical adjustment for better fit
    pdf.set_y(y_start + 70)
    
    # Row 3: Interception Side-On (100mm wide) + Crease Width Split (85mm wide)
    pdf.cell(0, 5, "5. INTERCEPTION SIDE-ON", 0, 0, 'L')
    pdf.set_x(105)
    pdf.cell(0, 5, "6. CREASE WIDTH SPLIT (SR)", 0, 1, 'L')
    y_start = pdf.get_y()
    pdf.image(fig_to_png_bytes(figures['seam_side_on']), x=10, y=y_start, w=100, type='PNG')
    pdf.image(fig_to_png_bytes(figures['seam_crease_split']), x=105, y=y_start + 10, w=85, type='PNG')
    pdf.set_y(y_start + 40) # Space for the side-on plot
    
    # Page 2: Scoring and Directional
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(213, 34, 33) 
    pdf.cell(0, 10, "SEAM SCORING & DIRECTIONAL ANALYSIS", 0, 1, 'L')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # Row 4: Top-On (90mm wide) + Wagon Wheel (90mm wide)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(95, 5, "7. INTERCEPTION TOP-ON", 0, 0, 'L')
    pdf.cell(95, 5, "8. SCORING AREAS (WAGON WHEEL)", 0, 1, 'L')
    y_start = pdf.get_y()
    pdf.image(fig_to_png_bytes(figures['seam_top_on']), x=10, y=y_start, w=95, type='PNG')
    pdf.image(fig_to_png_bytes(figures['seam_wagon_wheel']), x=105, y=y_start, w=95, type='PNG')
    pdf.set_y(y_start + 85) 
    
    # Row 5: L/R Split (90mm wide) + Swing (90mm wide)
    pdf.cell(95, 5, "9. LEFT/RIGHT RUN SPLIT", 0, 0, 'L')
    pdf.cell(95, 5, "10. SWING DIRECTIONAL SPLIT (AVG)", 0, 1, 'L')
    y_start = pdf.get_y()
    pdf.image(fig_to_png_bytes(figures['seam_lr_split']), x=10, y=y_start, w=95, type='PNG')
    pdf.image(fig_to_png_bytes(figures['seam_swing']), x=105, y=y_start, w=95, type='PNG')
    pdf.set_y(y_start + 45) 
    
    # Row 6: Deviation
    pdf.cell(0, 5, "11. DEVIATION DIRECTIONAL SPLIT (AVG)", 0, 1, 'L')
    pdf.image(fig_to_png_bytes(figures['seam_deviation']), x=10, y=pdf.get_y(), w=95, type='PNG')

    # --- SPIN ANALYSIS (New Page) ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(33, 67, 213) # Deep Blue
    pdf.cell(0, 10, "SPIN ANALYSIS", 0, 1, 'L')
    pdf.set_text_color(0, 0, 0) # Reset color
    pdf.ln(2)

    # Note: Spin layout mirrors Seam layout using 'spin_' keys
    # Row 1: Zonal (100mm wide) + Pitch Run % (50mm wide)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(100, 5, "1. CREASE BEEHIVE ZONES (SR)", 0, 0, 'L')
    pdf.cell(50, 5, "2. PITCH LENGTH RUN %", 0, 1, 'L')
    y_start = pdf.get_y()
    pdf.image(fig_to_png_bytes(figures['spin_zonal']), x=10, y=y_start, w=100, type='PNG')
    pdf.image(fig_to_png_bytes(figures['spin_pitch_pct']), x=115, y=y_start, w=50, type='PNG')
    pdf.set_y(y_start + 75)
    
    # Row 2: Beehive (100mm wide) + Lateral Boxes (85mm wide)
    pdf.cell(0, 5, "3. CREASE BEEHIVE", 0, 0, 'L')
    pdf.set_x(105)
    pdf.cell(0, 5, "4. LATERAL PERFORMANCE BOXES", 0, 1, 'L')
    y_start = pdf.get_y()
    pdf.image(fig_to_png_bytes(figures['spin_beehive'], is_plotly=True), x=10, y=y_start, w=100, type='PNG')
    pdf.image(fig_to_png_bytes(figures['spin_lateral']), x=105, y=y_start + 5, w=85, type='PNG') 
    pdf.set_y(y_start + 70)
    
    # Final step: Return the PDF as binary data
    return pdf.output(dest='S').encode('latin-1')

# =========================================================

# --- 3. MAIN STREAMLIT APP STRUCTURE (Modified to generate and use figures dict) ---

st.set_page_config(layout="wide")

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
    missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
    if missing_cols:
        st.error(f"The CSV file is missing required columns: {', '.join(missing_cols)}. Please ensure the correct data columns are present.")
        st.stop()
    
    # Data separation
    df_seam_raw = df_raw[df_raw["DeliveryType"] == "Seam"].copy()
    df_spin_raw = df_raw[df_raw["DeliveryType"] == "Spin"].copy()

    # --- Filters ---
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4) 

    all_teams = ["All"] + sorted(df_raw["BattingTeam"].dropna().unique().tolist())
    with filter_col1:
        bat_team = st.selectbox("Batting Team", all_teams, index=0)

    if bat_team != "All":
        batsmen_options = ["All"] + sorted(df_raw[df_raw["BattingTeam"] == bat_team]["BatsmanName"].dropna().unique().tolist())
    else:
        batsmen_options = ["All"] + sorted(df_raw["BatsmanName"].dropna().unique().tolist())
    
    with filter_col2:
        batsman = st.selectbox("Batsman Name", batsmen_options, index=0)

    innings_options = ["All"] + sorted(df_raw["Innings"].dropna().unique().tolist())
    with filter_col3:
        selected_innings = st.selectbox("Innings", innings_options, index=0)
    
    bowler_hand_options = ["All", "Right Hand", "Left Hand"]
    with filter_col4:
        selected_bowler_hand = st.selectbox("Bowler Hand", bowler_hand_options, index=0)
    
    # --- Apply Filters ---
    def apply_filters(df):
        if bat_team != "All":
            df = df[df["BattingTeam"] == bat_team]
        if batsman != "All":
            df = df[df["BatsmanName"] == batsman]
        if selected_innings != "All":
            df = df[df["Innings"] == selected_innings]
        if selected_bowler_hand != "All":
            is_right = (selected_bowler_hand == "Right Hand")
            df = df[df["IsBowlerRightHanded"] == is_right]
        return df

    df_seam = apply_filters(df_seam_raw)
    df_spin = apply_filters(df_spin_raw)
    
    # =========================================================
    # **CHART GENERATION DICTIONARY (CRUCIAL FOR PDF/DISPLAY)**
    # =========================================================
    figures = {
        'seam_zonal': create_zonal_analysis(df_seam, batsman, "Seam"),
        'seam_beehive': create_crease_beehive(df_seam, "Seam"),
        'seam_lateral': create_lateral_performance_boxes(df_seam, "Seam", batsman),
        'seam_pitchmap': create_pitch_map(df_seam, "Seam"),
        'seam_pitch_pct': create_pitch_length_run_pct(df_seam, "Seam"),
        'seam_side_on': create_interception_side_on(df_seam, "Seam"),
        'seam_crease_split': create_crease_width_split(df_seam, "Seam"),
        'seam_top_on': create_interception_front_on(df_seam, "Seam"),
        'seam_wagon_wheel': create_wagon_wheel(df_seam, "Seam"),
        'seam_lr_split': create_left_right_split(df_seam, "Seam"),
        'seam_swing': create_directional_split(df_seam, "Swing", "Swing", "Seam"),
        'seam_deviation': create_directional_split(df_seam, "Deviation", "Deviation", "Seam"),
        
        'spin_zonal': create_zonal_analysis(df_spin, batsman, "Spin"),
        'spin_beehive': create_crease_beehive(df_spin, "Spin"),
        'spin_lateral': create_lateral_performance_boxes(df_spin, "Spin", batsman),
        'spin_pitchmap': create_pitch_map(df_spin, "Spin"),
        'spin_pitch_pct': create_pitch_length_run_pct(df_spin, "Spin"),
        'spin_side_on': create_interception_side_on(df_spin, "Spin"),
        'spin_crease_split': create_crease_width_split(df_spin, "Spin"),
        'spin_top_on': create_interception_front_on(df_spin, "Spin"),
        'spin_wagon_wheel': create_wagon_wheel(df_spin, "Spin"),
        'spin_lr_split': create_left_right_split(df_spin, "Spin"),
        'spin_swing': create_directional_split(df_spin, "Swing", "Drift", "Spin"),
        'spin_deviation': create_directional_split(df_spin, "Deviation", "Turn", "Spin")
    }

    # --- ADD DOWNLOAD BUTTON ---
    heading_col, download_col = st.columns([4, 1])
    
    heading_text = batsman.upper() if batsman != "All" else "GLOBAL ANALYSIS"
    with heading_col:
        st.header(f"**{heading_text}**")
    
    with download_col:
        pdf_bytes = create_pdf_report(figures, heading_text)
        
        st.download_button(
            label="Download PDF Report ⬇️",
            data=pdf_bytes,
            file_name=f"Report_{heading_text.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
    # =========================================================

    # --- 4. DISPLAY CHARTS IN TWO COLUMNS (using the generated figures dictionary) ---
    
    col1, col2 = st.columns(2)
    
    # --- LEFT COLUMN: SEAM ANALYSIS ---
    with col1:
        st.markdown("#### SEAM")

        st.markdown("###### CREASE BEEHIVE ZONES")
        st.pyplot(figures['seam_zonal'], use_container_width=True)
        
        st.markdown("###### CREASE BEEHIVE")
        st.plotly_chart(figures['seam_beehive'], use_container_width=True)

        st.pyplot(figures['seam_lateral'], use_container_width=True)
        
        pitch_map_col, run_pct_col = st.columns([3, 1])

        with pitch_map_col:
            st.markdown("###### PITCHMAP")
            st.plotly_chart(figures['seam_pitchmap'], use_container_width=True)
            
        with run_pct_col:
            st.markdown("###### ")
            st.pyplot(figures['seam_pitch_pct'], use_container_width=True)
        
        st.markdown("###### INTERCEPTION SIDE-ON")
        st.pyplot(figures['seam_side_on'], use_container_width=True)

        st.pyplot(figures['seam_crease_split'], use_container_width=True)

        bottom_col_left, bottom_col_right = st.columns(2)

        with bottom_col_left:
            st.markdown("###### INTERCEPTION TOP-ON")
            st.pyplot(figures['seam_top_on'], use_container_width=True)
        
        with bottom_col_right:
            st.markdown("###### SCORING AREAS")    
            st.pyplot(figures['seam_wagon_wheel'], use_container_width=True)
            st.pyplot(figures['seam_lr_split'], use_container_width=True)
    
        final_col_swing, final_col_deviation = st.columns(2)

        with final_col_swing:
            st.pyplot(figures['seam_swing'], use_container_width=True)

        with final_col_deviation:
            st.pyplot(figures['seam_deviation'], use_container_width=True)   

    # --- RIGHT COLUMN: SPIN ANALYSIS ---
    with col2:
        st.markdown("#### SPIN")
        
        st.markdown("###### CREASE BEEHIVE ZONES")
        st.pyplot(figures['spin_zonal'], use_container_width=True)
        st.markdown("###### CREASE BEEHIVE")
        st.plotly_chart(figures['spin_beehive'], use_container_width=True)

        st.pyplot(figures['spin_lateral'], use_container_width=True)

        pitch_map_col, run_pct_col = st.columns([3, 1])

        with pitch_map_col:
            st.markdown("###### PITCHMAP")
            st.plotly_chart(figures['spin_pitchmap'], use_container_width=True) 
            
        with run_pct_col:
            st.markdown("###### ")
            st.pyplot(figures['spin_pitch_pct'], use_container_width=True)
        
        st.markdown("###### INTERCEPTION SIDE-ON")
        st.pyplot(figures['spin_side_on'], use_container_width=True)

        st.pyplot(figures['spin_crease_split'], use_container_width=True)

        bottom_col_left, bottom_col_right = st.columns(2)

        with bottom_col_left:
            st.markdown("###### INTERCEPTION TOP-ON")
            st.pyplot(figures['spin_top_on'], use_container_width=True)
        
        with bottom_col_right:
            st.markdown("###### SCORING AREAS")
            st.pyplot(figures['spin_wagon_wheel'], use_container_width=True)
            st.pyplot(figures['spin_lr_split'], use_container_width=True)
            
        final_col_swing, final_col_deviation = st.columns(2)

        with final_col_swing:
            st.pyplot(figures['spin_swing'], use_container_width=True)

        with final_col_deviation:
            st.pyplot(figures['spin_deviation'], use_container_width=True)    

else:
    st.info("⬆️ Please upload a CSV file to begin the analysis.")
