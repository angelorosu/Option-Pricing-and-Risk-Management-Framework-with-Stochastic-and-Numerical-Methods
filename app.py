import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your core models from core/ folder
from core.stochastic import GBMModel, HestonModel, VarianceGammaModel
from core.fixed_income import CIRModel
from core.pricing import BlackScholesPricing, MonteCarloPricing
from core.risk import VaRCalculator
from core.fixed_income import VasicekModel, CIRModel, BondPricing, YieldCurve

# Set up the main page
st.set_page_config(page_title="Finance Dashboard", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Stochastic Models", "Option Pricing", "Risk Analytics", "Fixed Income"])

# Sidebar for global parameters
st.sidebar.title("Global Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", min_value=1, value=100)
T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.1, value=1.0)
paths = st.sidebar.number_input("Number of Paths", min_value=10, value=1000)
steps = st.sidebar.number_input("Number of Steps", min_value=10, value=252)
run_button = st.sidebar.button("Run Simulation")

# Home Page (Dashboard)
if page == "Home":
    st.title("📊 Financial Analytics Dashboard")
    st.write("Welcome to the Financial Analytics Dashboard. Use the navigation on the left to explore different models, option pricing, and risk analytics.")
    st.markdown("### Key Metrics")
    st.write("Track important metrics like VaR, bond prices, and option prices here once they are computed from other sections.")

# Stochastic Models Page
elif page == "Stochastic Models":
    st.title("📈 Stochastic Models")
    model_type = st.selectbox("Select Model", ["GBM", "Heston", "CIR", "Variance Gamma"])
    if run_button:
        if model_type == "GBM":
            gbm = GBMModel(S0, mu=0.05, sigma=0.2, T=T, steps=steps, paths=paths)
            paths = gbm.generate_path()
        elif model_type == "Heston":
            heston = HestonModel(S0, T, steps, paths, v0=0.04, kappa=2.0, theta=0.04, rho=-0.7, mu=0.05)
            paths, _ = heston.generate_path()
        elif model_type == "CIR":
            cir = CIRModel(r0=0.03, a=0.1, b=0.04, sigma=0.01, T=T, steps=steps, paths=paths)
            paths = cir.simulate()
        elif model_type == "Variance Gamma":
            vg = VarianceGammaModel(S0, T, steps, paths, mu=0.05, sigma=0.2, kappa=0.1)
            paths = vg.generate_path()
        
        st.line_chart(paths.T)

# Option Pricing Page
elif page == "Option Pricing":
    st.title("💲 Option Pricing")
    option_type = st.selectbox("Option Type", ["Call", "Put"])
    pricing_method = st.selectbox("Pricing Method", ["Black-Scholes", "Monte Carlo"])
    K = st.number_input("Strike Price (K)", min_value=1, value=100)
    if run_button:
        if pricing_method == "Black-Scholes":
            bs = BlackScholesPricing(S0, K, r=0.05, T=T, sigma=0.2, option_type=option_type.lower())
            price = bs.price()
        elif pricing_method == "Monte Carlo":
            mc = MonteCarloPricing(S0, K, r=0.05, T=T, paths=paths, steps=steps, option_type=option_type.lower())
            # Reuse previously simulated paths from Stochastic Models
            price = mc.price(paths)
        st.write(f"Option Price: {price:.2f}")

# Risk Analytics Page
elif page == "Risk Analytics":
    st.title("📉 Risk Analytics")
    VaR_level = st.slider("Select VaR Confidence Level", min_value=0.01, max_value=0.10, step=0.01, value=0.05)
    if run_button:
        # Simulate returns
        returns = np.random.normal(0, 1, 1000)
        var_calculator = VaRCalculator(returns, alpha=VaR_level)
        var_value = var_calculator.calculate_var()
        st.write(f"Value-at-Risk (VaR) at {VaR_level*100}% level: {var_value:.2f}")

# Fixed Income Page
elif page == "Fixed Income":
    st.title("💸 Fixed Income Analytics")
    model_type = st.selectbox("Select Interest Rate Model", ["Vasicek", "CIR"])
    if run_button:
        if model_type == "Vasicek":
            vasicek = VasicekModel(r0=0.03, a=0.1, b=0.04, sigma=0.01, T=T, steps=steps, paths=paths)
            rates = vasicek.simulate()
        elif model_type == "CIR":
            cir = CIRModel(r0=0.03, a=0.1, b=0.04, sigma=0.01, T=T, steps=steps, paths=paths)
            rates = cir.simulate()

        st.line_chart(rates.T)
