# 📘 Usage Guide – Smart Demand & Trend Forecasting Agent

This guide covers setup, day-to-day usage, troubleshooting, and key design choices for the app.

---

## 🛠️ 1) Setup

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

Run the app:

```bash
streamlit run forecast_agent_app.py
```

---

## 📤 2) Upload & Configure

1. Upload a CSV/XLSX with:

   * One date/datetime column
   * One numeric metric to forecast
2. If detection is wrong, select the correct **Date** and **Target** columns from the dropdowns.
3. In the sidebar:

   * Choose frequency (auto / daily / weekly / monthly)
   * Set forecast horizon (days/weeks/months)
   * Pick insight model — default: `flan-t5-small`; fast fallback: `distilgpt2`
4. Review:

   * Forecast chart + confidence intervals
   * Metrics (MAPE, MAE, RMSE)
   * Seasonal decomposition (if series long enough)
5. Click **“Generate Insights”** to get short, plain-language commentary.
6. Download the **forecast CSV** and **insights text** as needed.

💡 *Tip:* For a demo dataset, run `python generate_time_data.py` to create `time_series_sales.csv`.

---

## ⚙️ 3) What the App Does

* Resamples the series to a tidy frequency and fills small gaps (interpolation).
* Fits a **Prophet** model to forecast the selected horizon.
* Computes in-sample metrics:

  * **MAPE** (lower is better)
  * **MAE** (mean absolute error)
  * **RMSE** (root mean square error)
* Shows an additive **seasonal decomposition** (trend, seasonal, residual).

---

## 🧩 4) Troubleshooting

**“Insights spinner keeps running”**

* Switch model to `distilgpt2` (light and fast).
* Stay on CPU mode; small GPUs can stall during model init.
* First run downloads models — cached afterward.

**“Decomposition not available”**

* Reduce seasonal period (e.g., 7 → 5 for daily data).
* Ensure the series covers several seasonal cycles.

**“Date not detected”**

* Manually select the column from the dropdown.
* Ensure date format is parseable (YYYY-MM-DD).

**Prophet install or compile issues**

* Upgrade pip: `python -m pip install --upgrade pip`
* Reinstall Prophet: `pip install prophet`
* On Windows, ensure Microsoft C++ build tools are installed.

**Transformers or torch issues**

* Stick to CPU builds; avoid GPU initialisation on small VRAM GPUs.
* Upgrade essentials:

  ```bash
  pip install --upgrade pip setuptools wheel
  ```
* Reinstall transformers cleanly:

  ```bash
  pip install --no-cache-dir transformers
  ```

**Port already in use**

* Run Streamlit on a different port:

  ```bash
  streamlit run forecast_agent_app.py --server.port 8502
  ```

---

## 🧠 5) Design Decisions (Summary)

* **Prophet as default:** Works well for business time-series with minimal tuning.
* **Resampling + gap fill:** Ensures continuity for accurate modeling.
* **Simple metrics:** MAPE/MAE/RMSE provide quick validation.
* **Local commentary:** Uses free LLMs (`flan-t5-small`, `distilgpt2`) — no API keys.
* **CPU-first:** Reliable on laptops, avoids GPU stalls, models cached for responsiveness.
* **Extensible design:** Easily expandable with ARIMA, regressors, holidays, backtesting, and DB connectors.

---

## 💡 6) Everyday Tips

* Keep uploaded files small and non-sensitive.
* For noisy series, use weekly resampling for smoother trends.
* Monthly data works best with longer horizons and fewer decomposition cycles.
* Save chart + insights screenshots for quick reporting.
* Cached models make repeated runs much faster.

