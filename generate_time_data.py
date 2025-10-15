import pandas as pd
import numpy as np

def create_synthetic_sales(path="time_series_sales.csv"):
    """Generate two years of synthetic daily sales data."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")

    # Trend + seasonality + noise
    trend = np.linspace(1000, 2000, len(dates))
    season = 200 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 70, len(dates))
    sales = trend + season + noise

    df = pd.DataFrame({"Date": dates, "Sales": sales})
    df.to_csv(path, index=False)
    print(f"✅  Synthetic data saved to {path}")

if __name__ == "__main__":
    create_synthetic_sales()