import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛒",
    layout="wide",
)

MODEL_PATH = "xgb_walmart_model.joblib"
DATA_PATH  = "Walmart.csv"

# ─────────────────────────────────────────────
# Helper: feature engineering (same as notebook)
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Year"]    = df["Date"].dt.year
    df["Month"]   = df["Date"].dt.month
    df["Day"]     = df["Date"].dt.day
    df["Week"]    = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df.drop("Date", axis=1, inplace=True)
    return df

FEATURE_COLS = [
    "Store", "Holiday_Flag", "Temperature",
    "Fuel_Price", "CPI", "Unemployment",
    "Year", "Month", "Day", "Week", "Quarter",
]

# ─────────────────────────────────────────────
# Train & cache model
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training XGBoost model…")
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        # load metrics if they were saved
        metrics = joblib.load("model_metrics.joblib") if os.path.exists("model_metrics.joblib") else None
        return model, metrics

    if not os.path.exists(DATA_PATH):
        return None, None

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df["Weekly_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2":   round(r2_score(y_test, y_pred), 4),
        "mae":  round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(metrics, "model_metrics.joblib")

    return model, metrics


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🛒 Walmart Weekly Sales Predictor")
st.caption("XGBoost model trained on Walmart store data  |  R² ≈ 0.98")

model, metrics = load_or_train_model()

# ── Model not available ──────────────────────
if model is None:
    st.error(
        "⚠️ **Model not found and `Walmart.csv` is missing.**\n\n"
        "Place `Walmart.csv` in the same folder as `app.py` and restart the app."
    )
    st.stop()

# ── Metrics row ──────────────────────────────
if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score",  metrics["r2"],  help="Closer to 1 is better")
    c2.metric("MAE",       f"${metrics['mae']:,.0f}", help="Mean Absolute Error")
    c3.metric("RMSE",      f"${metrics['rmse']:,.0f}", help="Root Mean Squared Error")
    st.divider()

# ─────────────────────────────────────────────
# Sidebar — Input Form
# ───────────────────────────────────pip install ──────────
with st.sidebar:
    st.header("📋 Input Parameters")
    st.markdown("Fill in the details below and click **Predict**.")

    store        = st.selectbox("Store Number", options=list(range(1, 46)), index=0)
    holiday_flag = st.radio("Holiday Week?", options=[0, 1],
                            format_func=lambda x: "Yes 🎉" if x == 1 else "No",
                            horizontal=True)
    date_input   = st.date_input("Week Start Date", value=pd.Timestamp("2023-01-06"))

    st.subheader("Economic Indicators")
    temperature  = st.number_input("Temperature (°F)", value=60.0, step=0.1, format="%.2f")
    fuel_price   = st.number_input("Fuel Price ($)",   value=3.50, step=0.01, min_value=0.0, format="%.3f")
    cpi          = st.number_input("CPI",              value=211.0, step=0.1, format="%.4f")
    unemployment = st.number_input("Unemployment (%)", value=7.5, step=0.1, min_value=0.0, format="%.3f")

    predict_btn = st.button("🔮 Predict Sales", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if predict_btn:
    ts = pd.Timestamp(date_input)
    input_data = pd.DataFrame([{
        "Store":        store,
        "Holiday_Flag": holiday_flag,
        "Temperature":  temperature,
        "Fuel_Price":   fuel_price,
        "CPI":          cpi,
        "Unemployment": unemployment,
        "Year":         ts.year,
        "Month":        ts.month,
        "Day":          ts.day,
        "Week":         int(ts.isocalendar()[1]),
        "Quarter":      ts.quarter,
    }])

    prediction = model.predict(input_data[FEATURE_COLS])[0]

    st.subheader("📊 Prediction Result")
    st.success(f"**Estimated Weekly Sales : ${prediction:,.2f}**")

    # ── Gauge chart ─────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prediction,
        title = {"text": "Weekly Sales ($)", "font": {"size": 18}},
        number= {"prefix": "$", "valueformat": ",.0f"},
        gauge = {
            "axis": {"range": [0, 3_000_000], "tickformat": ",.0f"},
            "bar":  {"color": "#1f77b4"},
            "steps": [
                {"range": [0,          1_000_000], "color": "#eef"},
                {"range": [1_000_000,  2_000_000], "color": "#bbf"},
                {"range": [2_000_000,  3_000_000], "color": "#88d"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": prediction,
            },
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Input Summary ───────────────────────
    with st.expander("🧾 View Input Summary"):
        display_df = input_data.copy()
        display_df.insert(0, "Predicted_Sales", round(prediction, 2))
        st.dataframe(display_df, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────
st.subheader("📌 Feature Importance (XGBoost)")

importance_df = pd.DataFrame({
    "Feature":   FEATURE_COLS,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=True)

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Blues",
    title="Which features drive predictions the most?",
)
fig_imp.update_layout(
    showlegend=False,
    coloraxis_showscale=False,
    height=420,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────
# Batch Prediction (CSV Upload)
# ─────────────────────────────────────────────
st.divider()
st.subheader("📂 Batch Prediction (Upload CSV)")
st.caption(
    "Upload a CSV with columns: `Store, Date, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment`"
)

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded:
    try:
        batch_df = pd.read_csv(uploaded)
        batch_df = engineer_features(batch_df)
        missing = [c for c in FEATURE_COLS if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            batch_df["Predicted_Weekly_Sales"] = model.predict(batch_df[FEATURE_COLS]).round(2)
            st.success(f"✅ Predicted {len(batch_df)} rows.")
            st.dataframe(batch_df[FEATURE_COLS + ["Predicted_Weekly_Sales"]], use_container_width=True)

            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown(
    "<br><hr><center style='color:grey;font-size:0.8rem;'>"
    "Built with Streamlit · XGBoost · Plotly"
    "</center>",
    unsafe_allow_html=True,
)