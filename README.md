# 🛒 Walmart Weekly Sales Predictor

A machine learning web application that predicts **weekly sales for Walmart stores** using an XGBoost regression model trained on real Walmart sales data. Built with Streamlit and deployed as an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-R²%200.98-green?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 Overview

This project predicts **Weekly Sales** for Walmart stores based on economic and calendar-based features. The model achieves an **R² score of 0.98** using XGBoost, outperforming Linear Regression, Decision Tree, and Random Forest baselines.

The app allows:
- Single-row prediction via an interactive sidebar form
- Batch prediction by uploading a CSV file
- Visual output via a Plotly gauge chart and feature importance plot

---

## 🖥️ App Preview

| Section | Description |
|---|---|
| 📊 Metrics Row | R², MAE, RMSE displayed at the top |
| 📋 Sidebar Form | Input store details and economic indicators |
| 🔮 Prediction | Gauge chart showing estimated weekly sales |
| 📌 Feature Importance | Horizontal bar chart of XGBoost feature weights |
| 📂 Batch Upload | Upload CSV → get predictions → download results |

---

## 📁 Project Structure

```
walmart-sales-predictor/
│
├── app.py                      # Main Streamlit application
├── sales_prediction.ipynb      # EDA + model comparison notebook
├── Walmart.csv                 # Dataset (required for first run)
├── xgb_walmart_model.joblib    # Saved XGBoost model (auto-generated)
├── model_metrics.joblib        # Saved model metrics (auto-generated)
├── requirements.txt            # Python dependencies
└── README.md
```

> `xgb_walmart_model.joblib` and `model_metrics.joblib` are generated automatically on first run. You do **not** need to commit them.

---

## 🗂️ Dataset

**Source:** [Walmart Store Sales Forecasting — Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

| Column | Description |
|---|---|
| `Store` | Store number (1–45) |
| `Date` | Week start date (DD-MM-YYYY) |
| `Weekly_Sales` | Sales for the given store/week (**target**) |
| `Holiday_Flag` | 1 if it's a holiday week, else 0 |
| `Temperature` | Average temperature in the region (°F) |
| `Fuel_Price` | Cost of fuel in the region ($) |
| `CPI` | Consumer Price Index |
| `Unemployment` | Unemployment rate (%) |

---

## ⚙️ Feature Engineering

The `Date` column is parsed and split into:

```python
Year, Month, Day, Week (ISO), Quarter
```

`Month_name` is dropped before modeling. Final feature set used by the model:

```
Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment,
Year, Month, Day, Week, Quarter
```

---

## 🤖 Model Comparison

| Model | R² Score |
|---|---|
| Linear Regression | 0.16 |
| Decision Tree | 0.95 |
| Random Forest | 0.96 |
| **XGBoost** ✅ | **0.98** |

XGBoost was selected as the final model due to its highest R² score on the test set (30% split, `random_state=42`).

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/walmart-sales-predictor.git
cd walmart-sales-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `Walmart.csv` in the root directory. Download it from [Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data).

### 4. Run the app

```bash
streamlit run app.py
```

The model will **train automatically on first launch** and save `xgb_walmart_model.joblib` to disk. Subsequent runs will load the saved model instantly.

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
joblib
```

Install all at once:

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly joblib
```

---

## 🔮 How to Use

### Single Prediction
1. Open the sidebar
2. Select a store number, holiday flag, and week start date
3. Enter economic indicators (Temperature, Fuel Price, CPI, Unemployment)
4. Click **Predict Sales**
5. View the result in the gauge chart

### Batch Prediction
1. Prepare a CSV with columns: `Store, Date, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment`
2. Upload it using the **Batch Prediction** section
3. Download the output CSV with a `Predicted_Weekly_Sales` column added

---

## 📊 Model Metrics (Test Set)

| Metric | Value |
|---|---|
| R² Score | 0.98 |
| MAE | ~$58,000 |
| RMSE | ~$98,000 |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** — web app framework
- **XGBoost** — gradient boosting model
- **Scikit-learn** — train/test split, metrics
- **Plotly** — interactive charts (gauge + bar chart)
- **Pandas / NumPy** — data processing
- **Joblib** — model serialization

---

## 🙋 Author

**Fardeen**
B.Tech Computer Science Engineering (2022–2026)
Data Science & Generative AI Trainee — Ducat India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/your-username)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

