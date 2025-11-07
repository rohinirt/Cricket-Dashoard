import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from io import StringIO, BytesIO
import base64
import matplotlib.patheffects as pe
import plotly.io as pio # <-- CRITICAL FIX: Added for Plotly PDF export
from fpdf import FPDF 

# --- 1. GLOBAL UTILITY FUNCTIONS & CONSTANTS ---

# Required columns check
# CRITICAL FIX: Added "Swing", "Deviation", "Innings", and "IsBowlerRightHanded"
REQUIRED_COLS = [
    "BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", 
    "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", 
    "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", 
    "InterceptionZ", "InterceptionY", "Over", "Innings", "IsBowlerRightHanded",
    "Swing", "Deviation" 
]

# Function to convert figure to PNG bytes for PDF embedding
def fig_to_png_bytes(fig, is_plotly=False):
    """Converts a figure (Matplotlib or Plotly) to PNG bytes."""
    buf = BytesIO()
    if is_plotly:
        try:
            pio.write_image(fig, buf, format='png', scale=2) 
        except ValueError:
            print("Error: Could not use pio.write_image. Ensure Kaleido is installed.")
            return None
    else:
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf

# --- PDF Generation Function ---
def create_pdf_report(figures, analysis_type, name):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Setup
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Performance Report: {name}", 0, 1, 'C')
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"{analysis_type} ANALYSIS", 0, 1, 'L')
    
    # Chart Inclusion Logic
    count = 1
    for key, fig in figures.items():
        if isinstance(fig, (go.Figure, plt.Figure)):
            is_plotly = isinstance(fig, go.Figure)
            fig_bytes = fig_to_png_bytes(fig, is_plotly)
            
            if fig_bytes is None:
                continue

            caption_parts = key.split('_')
            delivery = caption_parts[0].upper()
            chart_name = " ".join(caption_parts[1:]).replace('SPLIT', 'SPLIT -').replace('MAP', ' MAP').upper()
            
            pdf.set_font("Arial", '', 10)
            
            if pdf.get_y() + 75 > pdf.h - 15 and count > 1: 
                 pdf.add_page()
                 pdf.set_font("Arial", 'B', 14)
                 pdf.cell(0, 10, f"{delivery} ANALYSIS (Cont.)", 0, 1, 'L')

            pdf.cell(0, 5, f"{count}. {chart_name} ({delivery})", 0, 1, 'L')
            pdf.image(fig_bytes, w=90, h=70, type='PNG') 
            pdf.ln(5)
            
            count += 1

    return pdf.output(dest='S').encode('latin-1')

# --- CHART 1: ZONAL ANALYSIS (CBH Boxes) ---
def create_zonal_analysis(df_in, batsman_name, delivery_type):
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig

    is_right_handed = True
    handed_data = df_in["IsBatsmanRightHanded"].dropna().unique()
    if len(handed_data) > 0 and batsman_name != "All": is_right_handed = handed_data[0]

    fig, ax = plt.subplots(figsize=(6, 8))
    
    stump_zone_width = 0.5 
    zones_y = [-np.inf, -stump_zone_width, 0, stump_zone_width, np.inf] 
    zones_z = [-np.inf, 0.5, 1.0, np.inf] 

    df_in['ZoneY'] = pd.cut(df_in['StumpsY'], bins=zones_y, labels=[0, 1, 2, 3], include_lowest=True, right=False)
    df_in['ZoneZ'] = pd.cut(df_in['StumpsZ'], bins=zones_z, labels=[0, 1, 2], include_lowest=True, right=False)
    df_in['Zone'] = df_in['ZoneY'].astype(str) + '-' + df_in['ZoneZ'].astype(str)
    
    df_in = df_in[df_in['ZoneY'].isin([0, 1, 2])]

    zone_stats = df_in.groupby('Zone').agg(
        total_runs=('Runs', 'sum'),
        total_balls=('Runs', 'count'),
        wickets=('Wicket', 'sum')
    ).reset_index()

    zone_stats['StrikeRate'] = np.where(zone_stats['total_balls'] > 0, (zone_stats['total_runs'] / zone_stats['total_balls']) * 100, 0)
    zone_stats['Average'] = np.where(zone_stats['wickets'] > 0, zone_stats['total_runs'] / zone_stats['wickets'], np.inf)

    zone_map = {
        '0-0': (0, 2), '0-1': (0, 1), '0-2': (0, 0), 
        '1-0': (1, 2), '1-1': (1, 1), '1-2': (1, 0), 
        '2-0': (2, 2), '2-1': (2, 1), '2-2': (2, 0)  
    }
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title(f"Crease Beehive Zones: {delivery_type}", fontsize=16, weight='bold')

    for i in range(1, 3):
        ax.axvline(i, color='gray', linestyle='--', linewidth=0.5)
        ax.axhline(i, color='gray', linestyle='--', linewidth=0.5)

    min_sr = 0
    max_sr = zone_stats['StrikeRate'].max() if not zone_stats['StrikeRate'].empty and zone_stats['StrikeRate'].max() > 0 else 200
    norm = mcolors.Normalize(vmin=min_sr, vmax=max_sr)
    cmap = cm.get_cmap('YlOrRd')

    for index, row in zone_stats.iterrows():
        zone_key = row['Zone']
        if zone_key in zone_map:
            col, row_map = zone_map[zone_key]
            
            x_start = col 
            y_start = row_map 

            color = cmap(norm(row['StrikeRate']))
            
            rect = patches.Rectangle((x_start, y_start), 1, 1, facecolor=color, alpha=0.7, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            avg_text = f"{row['Average']:.0f}" if row['Average'] != np.inf else 'N/A'
            text_content = f"R: {row['total_runs']}\nB: {row['total_balls']}\nW: {row['wickets']}\nSR: {row['StrikeRate']:.0f}\nA: {avg_text}"
            
            ax.text(x_start + 0.5, y_start + 0.5, 
                    text_content, 
                    ha='center', va='center', 
                    fontsize=8, 
                    color='black', weight='bold')
    
    ax.text(0.5, -0.1, "Leg/Off Side", ha='center', fontsize=9)
    ax.text(1.5, -0.1, "Stumps", ha='center', fontsize=9)
    ax.text(2.5, -0.1, "Off/Leg Side", ha='center', fontsize=9)
    ax.text(-0.1, 0.5, "High", va='center', rotation=90, fontsize=9)
    ax.text(-0.1, 1.5, "Mid", va='center', rotation=90, fontsize=9)
    ax.text(-0.1, 2.5, "Low", va='center', rotation=90, fontsize=9)


    plt.tight_layout()
    return fig

# --- CHART 10: DIRECTIONAL SPLIT (SWING/DEVIATION) ---
def create_directional_split(df_in, column_name, display_name, delivery_type):
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    
    if column_name not in df_in.columns:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, f"Missing '{column_name}' column", ha='center', va='center'); ax.axis('off'); return fig
        
    df_in['Direction'] = np.where(df_in[column_name] < 0, 'LEFT', 'RIGHT')
    
    dir_stats = df_in.groupby('Direction').agg(
        total_runs=('Runs', 'sum'),
        wickets=('Wicket', 'sum'),
        total_balls=('Runs', 'count'),
        avg_movement=(column_name, 'mean')
    ).reset_index()

    dir_stats['StrikeRate'] = np.where(dir_stats['total_balls'] > 0, (dir_stats['total_runs'] / dir_stats['total_balls']) * 100, 0)
    
    left_data = dir_stats[dir_stats['Direction'] == 'LEFT'].iloc[0] if 'LEFT' in dir_stats['Direction'].values else pd.Series({'total_balls': 0, 'StrikeRate': 0, 'avg_movement': 0, 'wickets': 0, 'total_runs': 0})
    right_data = dir_stats[dir_stats['Direction'] == 'RIGHT'].iloc[0] if 'RIGHT' in dir_stats['Direction'].values else pd.Series({'total_balls': 0, 'StrikeRate': 0, 'avg_movement': 0, 'wickets': 0, 'total_runs': 0})

    total_balls = dir_stats['total_balls'].sum()
    
    fig_dir, ax_dir = plt.subplots(figsize=(6, 4))
    chart_title = f"{display_name.upper()} SPLIT ({delivery_type.upper()})"
    
    categories = [display_name]
    directions = ['RIGHT', 'LEFT']
    
    right_balls = right_data['total_balls']
    left_balls = left_data['total_balls']
    
    bar_data = [right_balls, -left_balls]
    
    x_limit = max(abs(bar_data[0]), abs(bar_data[1]))
    x_limit = x_limit * 1.2 if x_limit > 0 else 10 
    
    bars = ax_dir.barh(categories, bar_data, height=0.5, color=['#4CAF50', '#2196F3'], zorder=2) 

    ax_dir.set_xlim(-x_limit, x_limit)
    ax_dir.axvline(0, color='black', linewidth=1.5, zorder=3)
    ax_dir.set_xticks([])
    ax_dir.set_yticks([])
    
    for i, bar in enumerate(bars):
        bar_value = abs(bar_data[i])
        data = right_data if i == 0 else left_data
        
        if bar_value == 0 or total_balls == 0:
            continue
            
        percent = (bar_value / total_balls) * 100
        sr = data['StrikeRate']
        movement = data['avg_movement']
        
        label = f"{directions[i]}\n{percent:.0f}%\nSR: {sr:.0f}\nAvg Move: {movement:.2f}"
        
        bar_end_x = bar.get_x() + bar.get_width()
        padding = 0.05 * x_limit 

        if directions[i] == 'LEFT':
            text_x = bar_end_x + padding 
            ha_align = 'left' 
        else:
            text_x = bar_end_x - padding 
            ha_align = 'right' 

        ax_dir.text(text_x, 
                    bar.get_y() + bar.get_height() / 2, 
                    label,
                    ha=ha_align, va='center', 
                    fontsize=12, 
                    color='black', weight='bold') 

    ax_dir.set_title(chart_title, fontsize=14, weight='bold', color='black', pad=10)
    
    ax_dir.spines['top'].set_visible(False)
    ax_dir.spines['bottom'].set_visible(False) 
    ax_dir.spines['left'].set_visible(False)
    ax_dir.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig_dir

# --- PLACEHOLDERS for remaining chart functions ---
# NOTE: You MUST replace these placeholders with your actual function logic.
def create_lateral_performance_boxes(df_in, delivery_type, batsman_name):
    if df_in.empty: fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"{delivery_type} Lateral Boxes Placeholder", ha='center', va='center', fontsize=12); ax.axis('off'); return fig

def create_crease_beehive(df_in, delivery_type):
    if df_in.empty: fig = go.Figure().add_annotation(text="No Data", x=0.5, y=0.5, showarrow=False); return fig
    fig = go.Figure().update_layout(title=f"{delivery_type} Beehive Placeholder", height=300)
    return fig

def create_pitch_map(df_in, delivery_type):
    if df_in.empty: fig = go.Figure().add_annotation(text="No Data", x=0.5, y=0.5, showarrow=False); return fig
    fig = go.Figure().update_layout(title=f"{delivery_type} Pitch Map Placeholder", height=300)
    return fig

def create_pitch_length_run_pct(df_in, delivery_type):
    if df_in.empty: fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"{delivery_type} Length Run % Placeholder", ha='center', va='center', fontsize=12); ax.axis('off'); return fig

def create_interception_side_on(df_in, delivery_type):
    if df_in.empty: fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"{delivery_type} Interception Side-On Placeholder", ha='center', va='center', fontsize=12); ax.axis('off'); return fig

def create_crease_width_split(df_in, delivery_type):
    if df_in.empty: fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"{delivery_type} Crease Width Split Placeholder", ha='center', va='center', fontsize=12); ax.axis('off'); return fig

def create_interception_front_on(df_in, delivery_type):
    if df_in.empty: fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"{delivery_type} Interception Front-On Placeholder", ha='center', va='center', fontsize=12); ax.axis('off'); return fig

def create_wagon_wheel(df_in, delivery_type):
    if df_in.empty: fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"{delivery_type} Wagon Wheel Placeholder", ha='center', va='center', fontsize=12); ax.axis('off'); return fig

def create_left_right_split(df_in, delivery_type):
    if df_in.empty: fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"{delivery_type} Left/Right Split Placeholder", ha='center', va='center', fontsize=12); ax.axis('off'); return fig
