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
    "InterceptionZ", "InterceptionY", "Over", "Innings", "IsBowlerRightHanded", 
    "Swing", "Deviation" # Added columns used for filtering/plotting
]

# Function to encode Matplotlib figure to image for Streamlit (kept empty, no longer strictly needed)
def fig_to_image(fig):
    return fig

# --- PDF HELPERS ---

# Helper to convert any figure type to PNG bytes for FPDF
def fig_to_png_bytes(fig, is_plotly=False):
    """Converts a figure (Matplotlib or Plotly) to PNG bytes."""
    buf = BytesIO()
    if is_plotly:
        # NOTE: This requires the 'kaleido' library to be installed (pip install kaleido)
        # Kaleido is necessary to convert Plotly figures to static images.
        pio.write_image(fig, buf, format='png', width=600, height=450) 
    else:
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

# --- CHART FUNCTIONS (Placeholders, kept for structure) ---
# NOTE: All chart functions must return the figure object (plt.Figure or go.Figure)

# --- CHART 1: ZONAL ANALYSIS (CBH Boxes) ---
# ... (Function definition remains the same, returns plt.Figure)
def create_zonal_analysis(df_in, batsman_name, delivery_type):
    # ... (Zonal Analysis logic remains the same)
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(4, 3)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig

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
    cmap = cm.get_cmap('Reds')

    fig_boxes, ax = plt.subplots(figsize=(4,3), subplot_kw={'xticks': [], 'yticks': []}) 
    
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

        ax.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor="black", facecolor=color, linewidth=0.8))

        ax.text(x1 + w / 2, y1 + h / 2, 
        f"R: {runs}\nW: {wkts}\nSR: {sr:.0f}\nA: {avg:.0f}", 
        ha="center", 
        va="center", 
        fontsize=6, 
        color="black" if norm(avg) < 0.6 else "white", 
        linespacing=1.2)
    ax.set_title(f"STRIKE RATE", 
                 fontsize=8, 
                 weight='bold', 
                 pad=10)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(0, 2); ax.axis('off'); 
    plt.tight_layout(pad=0.5) 
    return fig_boxes
# ... (rest of chart functions are the same, removed for brevity)

# --- CHART 2a: CREASE BEEHIVE ---
def create_crease_beehive(df_in, delivery_type):
    # ... (returns go.Figure)
    if df_in.empty:
        return go.Figure().update_layout(title="No data for Beehive", height=400)

    # --- Data Filtering ---
    wickets = df_in[df_in["Wicket"] == True]
    non_wickets_all = df_in[df_in["Wicket"] == False]
    boundaries = non_wickets_all[(non_wickets_all["Runs"] == 4) | (non_wickets_all["Runs"] == 6)]
    regular_balls = non_wickets_all[(non_wickets_all["Runs"] != 4) & (non_wickets_all["Runs"] != 6)]
    fig_cbh = go.Figure()

    # 1. TRACE: Regular Balls
    fig_cbh.add_trace(go.Scatter(
        x=regular_balls["CreaseY"], y=regular_balls["CreaseZ"], mode='markers', name="Regular Ball",
        marker=dict(color='lightgrey', size=10, line=dict(width=1, color="white"), opacity=0.95)
    ))

    # 2. NEW TRACE: Boundary Balls
    fig_cbh.add_trace(go.Scatter(
        x=boundaries["CreaseY"], y=boundaries["CreaseZ"], mode='markers', name="Boundary",
        marker=dict(color='royalblue', size=12, line=dict(width=1, color="white"), opacity=0.95)
    ))

    # 3. TRACE: Wickets
    fig_cbh.add_trace(go.Scatter(
        x=wickets["CreaseY"], y=wickets["CreaseZ"], mode='markers', name="Wicket",
        marker=dict(color='red', size=12, line=dict(width=1, color="white"), opacity=0.95)
    ))

    # Stump lines & Crease lines
    fig_cbh.add_vline(x=-0.18, line=dict(color="grey", dash="dot", width=0.5)) 
    fig_cbh.add_vline(x=0.18, line=dict(color="grey", dash="dot", width=0.5))
    fig_cbh.add_vline(x=0, line=dict(color="grey", dash="dot", width=0.5))
    fig_cbh.add_vline(x=-0.92, line=dict(color="grey", width=0.5)) 
    fig_cbh.add_vline(x=0.92, line=dict(color="grey", width=0.5))
    fig_cbh.add_hline(y=0.78, line=dict(color="grey", width=0.5)) 
    fig_cbh.add_annotation(
        x=-1.5, y=0.78, text="Stump line", showarrow=False,
        font=dict(size=8, color="grey"), xanchor='left', yanchor='bottom'
    )
    fig_cbh.update_layout(
        height=300, 
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, 2], showgrid=False, zeroline=True, visible=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    
    return fig_cbh

# --- PDF COMPILER FUNCTION ---

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
    pdf.set_text_color(213, 34, 33) # Set color to a deep red for the section header
    pdf.cell(0, 10, "SEAM ANALYSIS", 0, 1, 'L')
    pdf.set_text_color(0, 0, 0) # Reset color to black

    # Chart 1: Zonal Analysis (100mm width, 75mm height)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, "1. CREASE BEEHIVE ZONES", 0, 1, 'L')
    pdf.image(fig_to_png_bytes(figures['seam_zonal']), x=10, y=pdf.get_y(), w=100, type='PNG')
    pdf.set_y(pdf.get_y() + 75)
    
    # Chart 2: Crease Beehive (Plotly) (100mm width, 70mm height)
    pdf.cell(0, 5, "2. CREASE BEEHIVE", 0, 1, 'L')
    pdf.image(fig_to_png_bytes(figures['seam_beehive'], is_plotly=True), x=10, y=pdf.get_y(), w=100, type='PNG')
    pdf.set_y(pdf.get_y() + 70)
    
    # Charts 5 & 6 (Interception Top-On and Wagon Wheel) - Side by Side (Page 2)
    pdf.add_page()
    pdf.set_text_color(213, 34, 33) # Set color to a deep red
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "SEAM - Interception & Scoring", 0, 1, 'L')
    pdf.set_text_color(0, 0, 0) # Reset color to black
    
    # 5. Interception Top-On (90mm width)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, "5. INTERCEPTION TOP-ON", 0, 0, 'L')
    # 6. Wagon Wheel (90mm width, aligned next to 5)
    pdf.set_x(105) # Move to the right column
    pdf.cell(0, 5, "6. SCORING AREAS", 0, 1, 'L')
    
    # Images for 5 & 6
    y_start = pdf.get_y()
    pdf.image(fig_to_png_bytes(figures['seam_top_on']), x=10, y=y_start, w=90, type='PNG')
    pdf.image(fig_to_png_bytes(figures['seam_wagon_wheel']), x=105, y=y_start, w=90, type='PNG')
    pdf.set_y(y_start + 100) # Space for the two tall charts

    # --- SPIN ANALYSIS (New Page) ---
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(33, 67, 213) # Set color to a deep blue
    pdf.cell(0, 10, "SPIN ANALYSIS", 0, 1, 'L')
    pdf.set_text_color(0, 0, 0) # Reset color to black

    # Spin Zonal Analysis (100mm width)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, "1. CREASE BEEHIVE ZONES", 0, 1, 'L')
    pdf.image(fig_to_png_bytes(figures['spin_zonal']), x=10, y=pdf.get_y(), w=100, type='PNG')
    pdf.set_y(pdf.get_y() + 75)
    
    # Spin Crease Beehive (Plotly) (100mm width)
    pdf.cell(0, 5, "2. CREASE BEEHIVE", 0, 1, 'L')
    pdf.image(fig_to_png_bytes(figures['spin_beehive'], is_plotly=True), x=10, y=pdf.get_y(), w=100, type='PNG')
    pdf.set_y(pdf.get_y() + 70)
    
    # Final step: Return the PDF as binary data
    # NOTE: Output is bytes, which is what st.download_button expects
    return pdf.output(dest='S').encode('latin-1')

# ... (End of utility functions and PDF compiler)

# --- 3. MAIN STREAMLIT APP STRUCTURE ---

st.set_page_config(layout="wide")

# --- 3. MAIN STREAMLIT APP STRUCTURE ---
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
    
    # Initial validation and required column check (expanded for completeness)
    missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
    if missing_cols:
        st.error(f"The CSV file is missing required columns: {', '.join(missing_cols)}")
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
    
    # --- CHART GENERATION (Called once to populate figures for display and PDF) ---
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
        # Use a function wrapper (lambda) to delay PDF generation until button is clicked
        # Note: PDF generation can be slow, especially with many charts
        pdf_bytes = create_pdf_report(figures, heading_text)
        
        st.download_button(
            label="Download PDF Report ⬇️",
            data=pdf_bytes,
            file_name=f"Report_{heading_text.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )

    # --- 4. DISPLAY CHARTS IN TWO COLUMNS (using the generated figures) ---
    
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
