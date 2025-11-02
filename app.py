import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="Cricket Crease Beehive", layout="wide")
st.title("🏏 Crease Beehive Chart (Stump View)")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Hawkeye data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Required columns: {required_cols}")
    else:
        # --- Filters ---
        batsmen = st.multiselect("Select Batsman", options=df["BatsmanName"].unique())
        delivery_types = st.multiselect("Select Delivery Type", options=df["DeliveryType"].unique())
        
        filtered_df = df.copy()
        if batsmen:
            filtered_df = filtered_df[filtered_df["BatsmanName"].isin(batsmen)]
        if delivery_types:
            filtered_df = filtered_df[filtered_df["DeliveryType"].isin(delivery_types)]

        # --- Define Colors ---
        color_map = {True: 'red', False: 'grey'}
        filtered_df["Color"] = filtered_df["Wicket"].map(color_map)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(4, 5))  # smaller chart
        ax.scatter(
            filtered_df["StumpsY"],
            filtered_df["StumpsZ"],
            c=filtered_df["Color"],
            s=40,
            alpha=0.7,
            linewidths=0  # no boundary
        )

        # Vertical lines for stumps
        ax.axvline(x=-0.18, color='black', linestyle='--', linewidth=1.2)
        ax.axvline(x=0.18, color='black', linestyle='--', linewidth=1.2)

        # Fixed axes
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(0, 2.5)
        ax.set_xlabel("StumpsY (Left–Right)")
        ax.set_ylabel("StumpsZ (Height)")
        ax.set_title("Crease Beehive (Stump View)", fontsize=12)

        # Hide top/right borders for a clean look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to visualize the beehive chart.")
