````markdown
# 📈 Smart Demand & Trend Forecasting Agent

An interactive Streamlit app for quick, repeatable time-series analysis.  
Upload a CSV/XLSX, pick the date and metric, and the app will:

- Fit a Prophet forecast for the next N days/weeks/months  
- Show error metrics (MAPE, MAE, RMSE)  
- Visualize seasonal decomposition (when the series is long enough)  
- Generate short, plain-language commentary using a lightweight local model  
- Export the forecast to CSV  

Everything runs locally; no paid APIs are required.

---

## 🧭 Why this exists

The first pass on a new dataset is usually a rush: “What’s the trend? How stable is it? What should we expect next month?”  
This tool standardises that step so analysts can produce a reliable forecast and a tidy summary in minutes.

---

## ⚙️ Quick start

### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run forecast_agent_app.py
````

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run forecast_agent_app.py
```

---

## 📂 Input format

* One column must be a date/datetime (e.g., Date, day, timestamp).
* At least one numeric column to forecast (e.g., Sales, Revenue, Visits).
* If the app guesses wrong, you can override the columns in the UI.

---

## 📊 What you’ll see

* Interactive forecast with confidence intervals
* Three quick metrics (MAPE, MAE, RMSE) to sanity-check fit
* Optional seasonal decomposition plot
* Three short, practical commentary bullets
* Download buttons for forecast CSV and insights text

---

## 📁 Files

* `forecast_agent_app.py` — Streamlit app
* `forecasting_utils.py` — data prep, Prophet, metrics, decomposition
* `generate_time_data.py` — synthetic daily sales for quick tests
* `requirements.txt` — pinned package versions
* `Datasets/` — optional sample files (light and non-sensitive)

---

## 🧰 Tech stack

* **Python**, **pandas**, **plotly**, **Streamlit**
* **Prophet** for forecasting
* **statsmodels** for decomposition
* **transformers** for lightweight local text generation
* CPU-only by default for reliability on modest hardware



