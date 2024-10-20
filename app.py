"""
Entrypoint to Markowitz Optimization Apps
(c) 2024-10 Marek Ozana
"""

import streamlit as st

st.logo("data/icon.png")
pg = st.navigation(
    {
        "Markowitz Portfolio Optimization": [
            st.Page(
                "scripts/Markowitz.py",
                title="SEB Hybrid",
                default=True,
                icon=":material/finance:",
            ),
            st.Page(
                "scripts/Micro_Finance_Analyzer.py",
                title="SEB Micro Finance",
                icon=":material/payments:",
            ),
        ],
        "History": [
            st.Page(
                "scripts/Historical_Risk_Return.py",
                title="Risk/Return",
                icon=":material/data_exploration:",
            )
        ],
    }
)

pg.run()
