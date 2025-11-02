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
    
    # Ensure required columns exist
    required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Required: {required_cols}")
    else:
        # --- Filters ---
        batsmen = st.multiselect("Select Batsman", options=df["BatsmanName"].unique(), default=None)
        delivery_types = st.multiselect("Select Delivery Type", options=df["DeliveryType"].unique(), default=None)
        
        filtered_df = df.copy()
        if batsmen:
            filtered_df = filtered_df[filtered_df["BatsmanName"].isin(batsmen)]
        if delivery_types:
            filtered_df = filtered_df[filtered_df["DeliveryType"].isin(delivery_types)]
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(4, 6))
        scatter = ax.scatter(
            filtered_df["StumpsY"],
            filtered_df["StumpsZ"],
            c=filtered_df["Wicket"].astype("category").cat.codes,
            cmap="coolwarm",
            alpha=0.7,
            s=60,
            edgecolor="black"
        )
        
        # Vertical lines for stumps
        ax.axvline(x=-0.18, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(x=0.18, color='black', linestyle='--', linewidth=1.5)
        
        # Axis limits
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(0, 2.5)
        ax.set_xlabel("StumpY (Left–Right)")
        ax.set_ylabel("StumpZ (Height)")
        ax.set_title("Crease Beehive (Stump View)")
        
        # Add legend
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Wicket (encoded)")
        
        st.pyplot(fig)
else:
    st.info("Upload a CSV file to begin.")
