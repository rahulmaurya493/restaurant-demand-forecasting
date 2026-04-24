import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="🍽️ Restaurant Demand Forecasting",
    page_icon="🍽️",
    layout="centered"
)

# ─── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('best_model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# ─── Feature Engineering ────────────────────────────────────
def create_features(date, lag1, lag7, lag30, 
                    roll7, roll14, roll30, roll7std):
    return pd.DataFrame({
        'day_of_week':   [date.weekday()],
        'day_of_month':  [date.day],
        'month':         [date.month],
        'quarter':       [date.quarter],
        'year':          [date.year],
        'is_weekend':    [1 if date.weekday() >= 5 else 0],
        'is_holiday':    [0],
        'lag_1':         [lag1],
        'lag_7':         [lag7],
        'lag_30':        [lag30],
        'rolling_7_mean':  [roll7],
        'rolling_14_mean': [roll14],
        'rolling_30_mean': [roll30],
        'rolling_7_std':   [roll7std]
    })

# ─── UI ─────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#ff6b35;'>
    🍽️ Restaurant Demand Forecasting
    </h1>
    <p style='text-align:center; color:gray;'>
    AI-powered daily sales prediction system
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Model Results Section ──────────────────────────────────
st.markdown("### 📊 Model Performance")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🏆 Best Model", "XGBoost")
with col2:
    st.metric("📉 MAE", "3.83")
with col3:
    st.metric("📉 RMSE", "4.75")

st.markdown("---")

# ─── Prediction Section ─────────────────────────────────────
st.markdown("### 🔮 Predict Sales for a Date")

col1, col2 = st.columns(2)

with col1:
    date_input = st.date_input("📅 Select Date:")
    lag1 = st.number_input("Yesterday's Sales (lag_1):", 
                            value=50, min_value=0)
    lag7 = st.number_input("Last Week Sales (lag_7):", 
                            value=48, min_value=0)
    lag30 = st.number_input("Last Month Sales (lag_30):", 
                             value=45, min_value=0)

with col2:
    roll7 = st.number_input("7-Day Average:", 
                             value=49.0, min_value=0.0)
    roll14 = st.number_input("14-Day Average:", 
                              value=48.0, min_value=0.0)
    roll30 = st.number_input("30-Day Average:", 
                              value=47.0, min_value=0.0)
    roll7std = st.number_input("7-Day Std Dev:", 
                                value=3.0, min_value=0.0)

predict_btn = st.button(
    "🔍 Predict Sales!", 
    use_container_width=True
)

# ─── Prediction Output ──────────────────────────────────────
if predict_btn:
    with st.spinner("Predicting... ⏳"):
        date = pd.Timestamp(date_input)
        features = create_features(
            date, lag1, lag7, lag30,
            roll7, roll14, roll30, roll7std
        )
        prediction = model.predict(features)[0]

    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Date", str(date_input))
    with col2:
        st.metric("📦 Predicted Sales", f"{prediction:.0f} units")
    with col3:
        day_name = date.day_name()
        is_weekend = "🎉 Weekend" if date.weekday() >= 5 else "💼 Weekday"
        st.metric("📆 Day Type", is_weekend)

    # Recommendation
    if prediction > 60:
        st.success(f"📈 HIGH demand expected! Stock up on ingredients!")
    elif prediction > 40:
        st.info(f"📊 MODERATE demand expected. Normal stock levels.")
    else:
        st.warning(f"📉 LOW demand expected. Reduce stock to avoid waste!")

# ─── Sales Trend Chart ──────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Sample Forecast vs Actual Chart")

# Sample data for visualization
dates = pd.date_range('2017-11-01', periods=30, freq='D')
np.random.seed(42)
actual = np.random.randint(35, 70, 30)
predicted = actual + np.random.randint(-5, 5, 30)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates, actual, label='Actual Sales', 
        color='blue', linewidth=2)
ax.plot(dates, predicted, label='XGBoost Forecast',
        color='red', linestyle='--', linewidth=2)
ax.fill_between(dates, actual, predicted, alpha=0.1, color='red')
ax.set_title('Actual vs Predicted Sales (Test Period)', fontsize=13)
ax.set_xlabel('Date')
ax.set_ylabel('Sales Units')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ─── Feature Importance ─────────────────────────────────────
st.markdown("---")
st.markdown("### 🔑 Feature Importance")

features_list = [
    'day_of_week', 'day_of_month', 'month',
    'quarter', 'year', 'is_weekend', 'is_holiday',
    'lag_1', 'lag_7', 'lag_30',
    'rolling_7_mean', 'rolling_14_mean',
    'rolling_30_mean', 'rolling_7_std'
]

importance_df = pd.DataFrame({
    'Feature': features_list,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(importance_df['Feature'], 
         importance_df['Importance'],
         color='steelblue', edgecolor='black')
ax2.set_title('XGBoost Feature Importance', fontsize=13)
ax2.set_xlabel('Importance Score')
plt.tight_layout()
st.pyplot(fig2)

# ─── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <p style='text-align:center; color:gray;'>
    Built by Rahul Maurya | Data Science Intern @ Infotact |
    Restaurant Demand Forecasting System
    </p>
""", unsafe_allow_html=True)
