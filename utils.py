import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- 1. GLOBAL UTILITY FUNCTIONS & CONSTANTS ---

# Simplified required columns for a quick trial
REQUIRED_COLS = [
    "BatsmanName", "DeliveryType", "BattingTeam", "Runs", "Innings", 
    "IsBowlerRightHanded"
]

# --- SIMPLE CHART 1: Bar Chart (Plotly) ---
def create_zonal_analysis(df_in, batsman_name, delivery_type):
    """Placeholder: Bar chart showing total runs scored."""
    if df_in.empty:
        return go.Figure().add_annotation(text="No Data", x=0.5, y=0.5, showarrow=False).update_layout(height=300)
    
    run_sum = df_in.groupby('BatsmanName')['Runs'].sum().reset_index()
    if run_sum.empty:
        return go.Figure().add_annotation(text="No Data", x=0.5, y=0.5, showarrow=False).update_layout(height=300)
    
    fig = px.bar(run_sum, x='BatsmanName', y='Runs', 
                 title=f"Total Runs ({delivery_type})",
                 height=350)
    fig.update_layout(title_font_size=16)
    return fig

# --- SIMPLE CHART 2: Pie Chart (Matplotlib) ---
def create_directional_split(df_in, column_name, display_name, delivery_type):
    """Placeholder: Pie chart showing balls count split by Innings."""
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Data", ha='center', va='center'); ax.axis('off'); return fig
    
    stats = df_in.groupby('Innings')['Runs'].count().reset_index()
    stats.columns = ['Innings', 'Balls']

    if stats.empty or stats['Balls'].sum() == 0:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.text(0.5, 0.5, "No Balls Recorded", ha='center', va='center'); ax.axis('off'); return fig

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Use Innings names for labels
    labels = [f"Innings {int(i)}" for i in stats['Innings']]
    
    ax.pie(stats['Balls'], 
           labels=labels, 
           autopct='%1.1f%%', 
           startangle=90, 
           textprops={'fontsize': 10, 'weight': 'bold'})
    
    ax.set_title(f"Balls Distribution by Innings ({delivery_type})", fontsize=12, weight='bold')
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.tight_layout()
    return fig


# --- PLACEHOLDERS (Returning generic figures) ---

# Matplotlib placeholders (Return object from plt.subplots())
def create_lateral_performance_boxes(df_in, delivery_type, batsman_name):
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Lateral Boxes Placeholder", ha='center', va='center'); ax.axis('off'); return fig
def create_pitch_length_run_pct(df_in, delivery_type):
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Length Run % Placeholder", ha='center', va='center'); ax.axis('off'); return fig
def create_interception_side_on(df_in, delivery_type):
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Interception Side-On Placeholder", ha='center', va='center'); ax.axis('off'); return fig
def create_crease_width_split(df_in, delivery_type):
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Crease Split Placeholder", ha='center', va='center'); ax.axis('off'); return fig
def create_interception_front_on(df_in, delivery_type):
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Interception Top-On Placeholder", ha='center', va='center'); ax.axis('off'); return fig
def create_wagon_wheel(df_in, delivery_type):
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Wagon Wheel Placeholder", ha='center', va='center'); ax.axis('off'); return fig
def create_left_right_split(df_in, delivery_type):
    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"L/R Split Placeholder", ha='center', va='center'); ax.axis('off'); return fig

# Plotly placeholders (Return object from go.Figure())
def create_crease_beehive(df_in, delivery_type):
    fig = go.Figure().update_layout(title=f"Beehive Placeholder", height=300); return fig
def create_pitch_map(df_in, delivery_type):
    fig = go.Figure().update_layout(title=f"Pitch Map Placeholder", height=300); return fig
