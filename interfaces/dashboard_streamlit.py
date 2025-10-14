# SPDX-License-Identifier: MIT
import pandas as pd
import streamlit as st

from core.indicators.entropy import delta_entropy, entropy
from core.indicators.kuramoto import compute_phase, kuramoto_order

st.title("TradePulse — Real-time Indicators")
uploaded = st.file_uploader("Upload CSV with columns: ts, price, volume", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    phases = compute_phase(df["price"].to_numpy())
    R = kuramoto_order(phases[-200:])
    H = entropy(df["price"].to_numpy()[-200:])
    dH = delta_entropy(df["price"].to_numpy(), window=200)
    st.metric("Kuramoto R", f"{R:.3f}")
    st.metric("Entropy H(200)", f"{H:.3f}")
    st.metric("ΔH(200)", f"{dH:.3f}")
    st.line_chart(df[["price"]])
