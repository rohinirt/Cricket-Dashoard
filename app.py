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
        # --- Filters ---
        batsmen = st.multiselect("Select Batsman", options=df["BatsmanName"].unique())
        delivery_types = st.multiselect("Select Delivery Type", options=df["DeliveryType"].unique())

        filtered_df = df.copy()
        if batsmen:
            filtered_df = filtered_df[filtered_df["BatsmanName"].isin(batsmen)]
        if delivery_types:
            filtered_df = filtered_df[filtered_df["DeliveryType"].isin(delivery_types)]

        # --- Separate by wicket ---
        wickets = filtered_df[filtered_df["Wicket"] == True]
        non_wickets = filtered_df[filtered_df["Wicket"] == False]

        # --- Create figure ---
        fig = go.Figure()

        # Non-wickets (grey)
        fig.add_trace(go.Scatter(
            x=non_wickets["StumpsY"],
            y=non_wickets["StumpsZ"],
            mode='markers',
            marker=dict(
                color='lightgrey',
                size=8,
                opacity=0.8,
                line=dict(width=0.8, color='white')  # white border
            ),
            name="No Wicket"
        ))

        # Wickets (red)
        fig.add_trace(go.Scatter(
            x=wickets["StumpsY"],
            y=wickets["StumpsZ"],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                opacity=0.9,
                line=dict(width=1, color='white')  # white border
            ),
            name="Wicket"
        ))

        # --- Stump lines ---
        fig.add_vline(x=-0.18, line=dict(color="black", dash="dot", width=1.2))
        fig.add_vline(x=0.18, line=dict(color="black", dash="dot", width=1.2))
        fig.add_vline(x=-0.92, line=dict(color="black", width=1.2))
        fig.add_vline(x=0.92, line=dict(color="black", width=1.2))

        # --- Colored stump zones ---
        fig.add_shape(type="rect", x0=-2.5, x1=-0.18, y0=0, y1=2.5,
                      fillcolor="rgba(0,255,0,0.05)", line_width=0)
        fig.add_shape(type="rect", x0=0.18, x1=2.5, y0=0, y1=2.5,
                      fillcolor="rgba(255,0,0,0.05)", line_width=0)

        # --- Layout ---
        fig.update_layout(
            width=750,
            height=400,
            xaxis=dict(
                range=[-1.6, 1.6],
                showgrid=True,
                zeroline=False,
                scaleanchor="y",
                scaleratio=1,
                showticklabels=False,  # hide tick labels
                title=None,  # hide title
            ),
            yaxis=dict(
                range=[0, 2.5],
                showgrid=True,
                zeroline=False,
                showticklabels=False,  # hide tick labels
                title=None,  # hide title
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            font=dict(size=12)
        )

        st.plotly_chart(fig, use_container_width=False)

else:
    st.info("👆 Upload a CSV file to view the crease beehive chart.")
