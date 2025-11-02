import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="Cricket Crease Beehive", layout="wide")
st.title("🏏 Crease Beehive Chart (Stump View)")

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload your Hawkeye data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Required columns
    required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpY", "StumpZ"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Required: {required_cols}")
    else:
        # --- Filters ---
        batsmen = st.multiselect("Select Batsman", options=df["BatsmanName"].unique())
        delivery_types = st.multiselect("Select Delivery Type", options=df["DeliveryType"].unique())
        
        filtered_df = df.copy()
        if batsmen:
            filtered_df = filtered_df[filtered_df["BatsmanName"].isin(batsmen)]
        if delivery_types:
            filtered_df = filtered_df[filtered_df["DeliveryType"].isin(delivery_types)]
        
        # --- Color Mapping for Wicket ---
        color_map = {True: "red", False: "grey", "True": "red", "False": "grey"}
        colors = filtered_df["Wicket"].map(color_map)
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(4, 5))  # smaller figure
        ax.scatter(
            filtered_df["StumpY"],
            filtered_df["StumpZ"],
            c=colors,
            alpha=0.8,
            s=40,
            edgecolors="none"  # no boundary
        )
        
        # Vertical lines for stumps
        ax.axvline(x=-0.18, color='black', linestyle='--', linewidth=1.2)
        ax.axvline(x=0.18, color='black', linestyle='--', linewidth=1.2)
        
        # Axis settings
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(0, 2.5)
        ax.set_xlabel("StumpY (Left–Right)")
        ax.set_ylabel("StumpZ (Height)")
        ax.set_title("Crease Beehive (Stump View)")
        
        # Cleaner layout
        ax.grid(False)
        ax.set_facecolor("white")
        st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to begin.")
