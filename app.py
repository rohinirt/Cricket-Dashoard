import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Cricket Crease Beehive", layout="wide")
st.title("🏏 Crease Beehive Chart (Stump View)")

uploaded_file = st.file_uploader("Upload your Hawkeye data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = ["BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Required columns: {required_cols}")
    else:
        # Filters
        batsmen = st.multiselect("Select Batsman", options=df["BatsmanName"].unique())
        delivery_types = st.multiselect("Select Delivery Type", options=df["DeliveryType"].unique())

        filtered_df = df.copy()
        if batsmen:
            filtered_df = filtered_df[filtered_df["BatsmanName"].isin(batsmen)]
        if delivery_types:
            filtered_df = filtered_df[filtered_df["DeliveryType"].isin(delivery_types)]

        # Split data for color logic
        wickets = filtered_df[filtered_df["Wicket"] == True]
        non_wickets = filtered_df[filtered_df["Wicket"] == False]

        # Create figure
        fig = go.Figure()

        # Non-wicket deliveries (grey)
        fig.add_trace(go.Scatter(
            x=non_wickets["StumpsY"],
            y=non_wickets["StumpsZ"],
            mode='markers',
            marker=dict(color='lightgrey', size=6, opacity=0.8),
            name="No Wicket"
        ))

        # Wicket deliveries (red)
        fig.add_trace(go.Scatter(
            x=wickets["StumpsY"],
            y=wickets["StumpsZ"],
            mode='markers',
            marker=dict(color='red', size=10, opacity=0.9),
            name="Wicket"
        ))

        # Add vertical stump lines
        fig.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1.5))
        fig.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1.5))

        # Layout adjustments
        fig.update_layout(
            width=450,
            height=400,
            xaxis=dict(title="StumpsY (Left–Right)", range=[-1.6, 1.6], zeroline=False, showgrid=True),
            yaxis=dict(title="StumpsZ (Height)", range=[0, 2.5], zeroline=False, showgrid=True),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            font=dict(size=12)
        )

        st.plotly_chart(fig, use_container_width=False)

else:
    st.info("👆 Upload a CSV file to view the crease beehive chart.")
