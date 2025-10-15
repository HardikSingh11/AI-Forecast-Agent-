import pandas as pd
from prophet import Prophet
import plotly.express as px
from typing import Tuple, Optional

def _infer_date_col(df: pd.DataFrame) -> Optional[str]:
    # prefer already-parsed datetime columns
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # try parse candidates quickly
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    return None

def _infer_target_col(df: pd.DataFrame) -> Optional[str]:
    nums = df.select_dtypes(include="number").columns.tolist()
    return nums[0] if nums else None

def prepare_data(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    target_col: Optional[str] = None,
    freq: str = "auto",
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Detect date/target columns, sort, resample if needed, and return Prophet-formatted df.
    freq: "auto" (try infer), "D" daily, "W" weekly, "M" monthly
    """
    df = df.copy()

    # detect columns
    date_col = date_col or _infer_date_col(df)
    if date_col is None:
        raise ValueError("No date-like column found.")
    target_col = target_col or _infer_target_col(df)
    if target_col is None:
        raise ValueError("No numeric target column found.")

    # parse & sort
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[[date_col, target_col]].dropna().rename(columns={date_col: "ds", target_col: "y"})
    df = df.sort_values("ds")

    # frequency inference / resample & fill gaps
    if freq == "auto":
        guessed = pd.infer_freq(df["ds"])
        # if cannot infer, assume daily
        freq = guessed if guessed else "D"

    # resample to tidy frequency
    df = df.set_index("ds").resample(freq).agg({"y": "mean"}).interpolate().reset_index()

    return df, date_col, target_col, freq

def build_forecast(df: pd.DataFrame, future_periods: int = 30):
    """Fit Prophet and return (model, forecast, plotly_figure)."""
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=future_periods, freq=pd.infer_freq(df["ds"]) or "D")
    forecast = model.predict(future)

    fig = px.line(
        forecast,
        x="ds",
        y=["yhat", "yhat_lower", "yhat_upper"],
        labels={"value": "Forecast", "ds": "Date"},
        title="Forecast with 95% Confidence Interval",
    )
    return model, forecast, fig

def in_sample_metrics(df: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """
    Compute simple in-sample metrics using forecast yhat on historical range.
    """
    hist = forecast[forecast["ds"] <= df["ds"].max()].merge(df, on="ds", how="inner")
    y_true = hist["y"].values
    y_pred = hist["yhat"].values
    eps = 1e-9
    mape = (abs((y_true - y_pred) / (y_true + eps))).mean() * 100.0
    mae = (abs(y_true - y_pred)).mean()
    rmse = ((y_true - y_pred) ** 2).mean() ** 0.5
    return {"MAPE": float(mape), "MAE": float(mae), "RMSE": float(rmse)}

# Optional: seasonal decomposition
def seasonal_decompose_figure(df: pd.DataFrame, period: int = 7):
    """
    Return a matplotlib figure of additive seasonal decomposition.
    Period: 7 for weekly seasonality (daily data), 12 for monthly (monthly data), etc.
    """
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose

    s = df.set_index("ds")["y"].asfreq(pd.infer_freq(df["ds"]) or "D").interpolate()
    result = seasonal_decompose(s, model="additive", period=period, two_sided=False, extrapolate_trend="freq")
    fig = result.plot()
    fig.set_size_inches(9.5, 7)
    plt.tight_layout()
    return fig