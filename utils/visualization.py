import numpy as np
import plotly.graph_objects as go

def generate_volatility_surface(strike_range=(50, 150), maturity_range=(0.1, 2.0), 
                                strike_step=5, maturity_step=0.1, 
                                base_vol=0.2, smile_factor=0.1, time_decay_factor=0.02):
    
    strikes = np.arange(strike_range[0], strike_range[1] + strike_step, strike_step)
    maturities = np.arange(maturity_range[0], maturity_range[1] + maturity_step, maturity_step)
    volatility_surface = np.zeros((len(maturities), len(strikes)))

    for i, t in enumerate(maturities):
        for j, k in enumerate(strikes):
            atm = 100  # Assume ATM strike is at 100
            volatility_surface[i, j] = base_vol + smile_factor * ((k - atm) / 100) ** 2 + time_decay_factor * t

    return strikes, maturities, volatility_surface


def plot_volatility_surface(strikes, maturities, volatility_surface):
    
    fig = go.Figure(
        data=[go.Surface(
            z=volatility_surface, 
            x=strikes, 
            y=maturities, 
            colorscale="Viridis"
        )]
    )
    
    fig.update_layout(
        title="Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price",
            yaxis_title="Time to Maturity",
            zaxis_title="Implied Volatility"
        )
    )
    return fig
