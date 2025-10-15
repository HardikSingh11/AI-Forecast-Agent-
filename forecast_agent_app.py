import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

from forecasting_utils import prepare_data, build_forecast, in_sample_metrics, seasonal_decompose_figure

st.set_page_config(page_title="📈 Smart Demand & Trend Forecasting Agent", layout="wide")
st.title("📊 Smart Demand & Trend Forecasting Agent")

# ------------------ LLM loader (cached) ------------------
@st.cache_resource(show_spinner=False)
def load_generator(model_key: str):
    device_map = "cpu"  # safest for 4GB VRAM
    dtype = torch.float32

    if model_key == "flan-t5-small":
        model_id = "google/flan-t5-small"
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)
        return pipeline("text2text-generation", model=mod, tokenizer=tok), "text2text-generation"

    if model_key == "distilgpt2":
        model_id = "distilgpt2"
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)
        return pipeline("text-generation", model=mod, tokenizer=tok), "text-generation"

    # optional
    if model_key == "phi-2":
        model_id = "microsoft/phi-2"
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)
        return pipeline("text-generation", model=mod, tokenizer=tok), "text-generation"

    # default
    model_id = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=dtype)
    return pipeline("text-generation", model=mod, tokenizer=tok), "text-generation"

def make_commentary(gen, task, prompt: str):
    if task == "text2text-generation":
        return gen(prompt, max_new_tokens=200)[0]["generated_text"]
    return gen(prompt, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]

# ------------------ Sidebar controls ------------------
with st.sidebar:
    st.header("Controls")
    freq_choice = st.selectbox("Data frequency", ["auto", "D (daily)", "W (weekly)", "M (monthly)"], index=0)
    freq_map = {"auto":"auto","D (daily)":"D","W (weekly)":"W","M (monthly)":"M"}
    freq = freq_map[freq_choice]

    horizon_unit = st.selectbox("Horizon unit", ["days", "weeks", "months"], index=0)
    horizon = st.slider("Forecast horizon", 7, 180, 30)
    if horizon_unit == "weeks":
        horizon_days = horizon * 7
    elif horizon_unit == "months":
        horizon_days = horizon * 30
    else:
        horizon_days = horizon

    model_choice = st.selectbox("Insight model", ["flan-t5-small", "distilgpt2", "phi-2"], index=0)

# ------------------ File upload ------------------
uploaded = st.file_uploader("📂 Upload a CSV or Excel with at least one Date column", type=["csv", "xlsx"])
if uploaded is None:
    st.info("👆 Upload a file to begin.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("📋 Data Preview")
st.dataframe(df_raw.head())

# Allow manual column override
date_candidates = [c for c in df_raw.columns if "date" in c.lower() or "time" in c.lower()] + df_raw.select_dtypes(include=["datetime"]).columns.tolist()
date_candidates = list(dict.fromkeys(date_candidates)) or df_raw.columns.tolist()
target_candidates = df_raw.select_dtypes(include="number").columns.tolist()

col1, col2 = st.columns(2)
with col1:
    date_sel = st.selectbox("Date column", options=date_candidates, index=0)
with col2:
    target_sel = st.selectbox("Target (numeric) column", options=target_candidates, index=0)

# ------------------ Prepare & Forecast ------------------
try:
    clean_df, date_col, target_col, used_freq = prepare_data(df_raw, date_col=date_sel, target_col=target_sel, freq=freq)
    st.caption(f"Using frequency: {used_freq}. Columns → Date: {date_col} | Target: {target_col}")
    model, forecast, fig = build_forecast(clean_df, future_periods=horizon_days)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Forecasting failed: {e}")
    st.stop()

# ------------------ Metrics & Decomposition ------------------
metrics = in_sample_metrics(clean_df, forecast)
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")
mcol2.metric("MAE", f"{metrics['MAE']:.2f}")
mcol3.metric("RMSE", f"{metrics['RMSE']:.2f}")

with st.expander("🧩 Seasonal Decomposition"):
    # choose period based on freq
    period_default = 7 if used_freq.startswith("D") else (12 if used_freq.startswith("M") else 4)
    period = st.number_input("Seasonal period", min_value=2, max_value=365, value=period_default, step=1)
    try:
        fig_dec = seasonal_decompose_figure(clean_df, period=period)
        st.pyplot(fig_dec)
    except Exception as e:
        st.warning(f"Decomposition not available: {e}")

# ------------------ Summary stats ------------------
mean_y = float(clean_df["y"].mean())
std_y = float(clean_df["y"].std())
hist_part = forecast[forecast["ds"] <= clean_df["ds"].max()]
growth = (hist_part["yhat"].iloc[-1] - hist_part["yhat"].iloc[-min(len(hist_part), horizon_days)]) / abs(hist_part["yhat"].iloc[-min(len(hist_part), horizon_days)]) * 100
st.info(f"Average value: {mean_y:.2f} | Std. Deviation: {std_y:.2f} | Predicted change next {horizon_days} days: {growth:.2f}%")

# ------------------ AI Commentary ------------------
st.subheader("🧠 AI‑Generated Commentary")
if st.button("Generate Insights"):
    with st.spinner("Analysing trends..."):
        stats_summary = f"Mean={mean_y:.2f}; Std={std_y:.2f}; Horizon={horizon_days}d; Expected change={growth:.2f}%; MAPE={metrics['MAPE']:.2f}%."

        prompt = f"""
You are a data analyst. Based on these figures:
{stats_summary}
Write 3 concise, actionable bullet points that:
- interpret the overall trend and variability,
- note likely seasonal implications,
- suggest one business action for the next {horizon_days} days.
Keep it brief and practical.
""".strip()

        try:
            gen, task = load_generator(model_choice)
            result = make_commentary(gen, task, prompt).strip()
            st.write(result)

            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(result.encode("utf-8"))
                tmp_path = tmp.name
            with open(tmp_path, "rb") as file:
                st.download_button("💾 Download Insights", data=file, file_name="forecast_insights.txt")
        except Exception as e:
            st.error(f"Insight generation failed: {e}")
            fallback = f"""
• Trend appears {'positive' if growth>0 else 'negative'} with a projected change of {abs(growth):.2f}% in {horizon_days} days.
• Variability (std={std_y:.2f}; MAPE={metrics['MAPE']:.2f}%) suggests watching volatility around seasonal periods.
• Prepare inventory/marketing for {'rising' if growth>0 else 'softening'} demand; review weekly seasonality.
""".strip()
            st.write(fallback)

# ------------------ Export forecast ------------------
with st.expander("⬇️ Downloads"):
    fcsv = forecast.rename(columns={"ds":"Date","yhat":"Forecast","yhat_lower":"Lower","yhat_upper":"Upper"})[["Date","Forecast","Lower","Upper"]]
    st.download_button("Download forecast CSV", data=fcsv.to_csv(index=False).encode("utf-8"), file_name="forecast.csv", mime="text/csv")