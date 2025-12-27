import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from arch import arch_model

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Quant Risk Analytics Platform", layout="wide")
st.title("Quant Risk Analytics Platform")

# -------------------- SIDEBAR --------------------
st.sidebar.header("Controls")

ticker = st.sidebar.text_input("Ticker", value="^NSEI")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))

rolling_window = st.sidebar.slider("Rolling Window (days)", 20, 252, 30)

confidence_level = st.sidebar.slider(
    "VaR Confidence Level", 0.90, 0.99, 0.95
)

shock_pct = st.sidebar.slider(
    "Stress Shock (%)", -20.0, -1.0, -5.0
) / 100

# -------------------- DATA LOADING --------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    return yf.download(ticker, start=start_date, auto_adjust=True)

data = load_data(ticker, start_date)

if data.empty:
    st.error("No data found. Check ticker symbol.")
    st.stop()

# -------------------- RETURNS --------------------
data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
returns = data["log_return"].dropna()

# -------------------- ROLLING VOLATILITY --------------------
data["rolling_vol"] = returns.rolling(rolling_window).std()

# -------------------- GARCH VOLATILITY --------------------
@st.cache_data
def fit_garch(returns):
    am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    garch_vol = res.conditional_volatility / 100
    return garch_vol, res

garch_vol, garch_result = fit_garch(returns)

# -------------------- RISK METRICS --------------------
hist_var = np.percentile(returns, (1 - confidence_level) * 100)
hist_es = returns[returns <= hist_var].mean()

mu, sigma = returns.mean(), returns.std()
gaussian_var = stats.norm.ppf(1 - confidence_level, mu, sigma)

# -------------------- METRICS --------------------
st.subheader("📌 Risk Metrics")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Historical VaR", f"{hist_var:.4%}")
c2.metric("Expected Shortfall", f"{hist_es:.4%}")
c3.metric("Gaussian VaR", f"{gaussian_var:.4%}")
c4.metric("Rolling Volatility", f"{data['rolling_vol'].iloc[-1]:.4%}")
c5.metric("GARCH Volatility", f"{garch_vol.iloc[-1]:.4%}")

# -------------------- RETURNS --------------------
st.subheader("📈 Daily Log Returns")
st.plotly_chart(px.line(returns, title="Daily Log Returns"),
                use_container_width=True)

# -------------------- VOLATILITY COMPARISON --------------------
st.subheader("📉 Volatility Comparison")

vol_df = pd.DataFrame({
    "Rolling Volatility": data["rolling_vol"],
    "GARCH Volatility": garch_vol
})

st.plotly_chart(
    px.line(vol_df, title="Rolling vs GARCH Volatility"),
    use_container_width=True
)

# -------------------- RETURN DISTRIBUTION --------------------
st.subheader("📊 Return Distribution & Tail Risk")

fig_hist = px.histogram(
    returns, nbins=100, marginal="box",
    title="Return Distribution"
)

fig_hist.add_vline(x=hist_var, line_dash="dash",
                   line_color="red", annotation_text="Historical VaR")

fig_hist.add_vline(x=gaussian_var, line_dash="dot",
                   line_color="black", annotation_text="Gaussian VaR")

st.plotly_chart(fig_hist, use_container_width=True)

# -------------------- QQ PLOT --------------------
st.subheader("📐 QQ Plot (Fat-Tail Diagnosis)")

theoretical_q, empirical_q = stats.probplot(returns, dist="norm")[0]

fig_qq = go.Figure()
fig_qq.add_trace(go.Scatter(
    x=theoretical_q, y=empirical_q,
    mode="markers", name="Empirical"
))
fig_qq.add_trace(go.Scatter(
    x=theoretical_q, y=theoretical_q,
    mode="lines", name="Normal Reference",
    line=dict(dash="dash")
))

st.plotly_chart(fig_qq, use_container_width=True)

# -------------------- NORMALITY TESTS --------------------
st.subheader("🧪 Normality Tests")

jb_stat, jb_p = stats.jarque_bera(returns)
sample_n = min(5000, len(returns))
sh_stat, sh_p = stats.shapiro(returns.sample(sample_n, random_state=42))

st.dataframe(pd.DataFrame({
    "Test": ["Jarque-Bera", "Shapiro-Wilk"],
    "Statistic": [jb_stat, sh_stat],
    "p-value": [jb_p, sh_p]
}))

# -------------------- ROLLING VAR --------------------
st.subheader("⏱️ Rolling Historical VaR")

rolling_var = returns.rolling(rolling_window).quantile(1 - confidence_level)

st.plotly_chart(
    px.line(rolling_var, title="Rolling Historical VaR"),
    use_container_width=True
)

# -------------------- DOWNSIDE RISK --------------------
st.subheader("📉 Downside Risk")

downside_returns = returns[returns < 0]
downside_vol = np.sqrt(downside_returns.var())

st.metric("Downside Volatility", f"{downside_vol:.4%}")

# -------------------- STRESS SCENARIO --------------------
st.subheader("🚨 Stress Scenario")

st.write(
    f"A **{shock_pct*100:.1f}%** shock corresponds to "
    f"~{shock_pct / returns.std():.1f} standard deviations "
    f"under historical volatility."
)

# -------------------- GARCH MODEL SUMMARY --------------------
with st.expander("📄 GARCH(1,1) Model Summary"):
    st.text(garch_result.summary())


