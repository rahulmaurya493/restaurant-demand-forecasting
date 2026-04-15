# 🍽️ Restaurant Demand Forecasting — AI/ML Project

## 📌 Project Overview
An AI-powered Demand Forecasting system that predicts 
daily restaurant sales using historical point-of-sale data.
Built as part of Infotact Technical Internship Program.

## 🎯 Problem Statement
Restaurants lose money due to over-ordering and under-ordering.
This model predicts future daily sales so managers can optimize
inventory and reduce food waste by 20-30%.

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn, Statsmodels

## 📁 Dataset
- Source: Store Item Demand Forecasting (Kaggle)
- Size: 9,13,000 rows
- Features: Date, Store, Item, Sales

## 🔬 Project Pipeline

### Week 1 — EDA
- Loaded and cleaned dataset
- Fixed datetime indexing
- Plotted overall sales trend
- Seasonality decomposition
- Autocorrelation analysis

### Week 2 — Feature Engineering
- Date features: day, month, quarter, year
- is_weekend and is_holiday flags
- Lag features: lag_1, lag_7, lag_30
- Rolling averages: 7, 14, 30 days
- Sequential train/test split (no data leakage)

### Week 3 — Model Training
- Baseline: Linear Regression
- Ensemble: Random Forest
- Advanced: XGBoost
- Time Series Cross Validation (5 folds)
- Hyperparameter Tuning

### Week 4 — Evaluation
- MAE and RMSE comparison
- Feature Importance analysis
- Final forecast visualization

## 📊 Model Results

| Model | MAE | RMSE |
|-------|-----|------|
| Linear Regression | 4.06 | 4.98 |
| Random Forest | 3.93 | 4.77 |
| XGBoost (baseline) | 3.90 | 4.87 |
| XGBoost (tuned) ✅ | 3.83 | 4.75 |

## 🏆 Best Model: XGBoost (Tuned)
- MAE: 3.83 (wrong by only ~4 sales units)
- Best Params: max_depth=4, learning_rate=0.05, n_estimators=100

## 🔑 Key Insights
- Lag features (lag_7) were strongest predictors
- Rolling averages smoothed noise significantly  
- Weekend and holiday flags improved accuracy
- Hyperparameter tuning reduced MAE by 0.07

## 📂 Repository Structure
restaurant-demand-forecasting/
│
├── restaurant_forecasting.ipynb  ← Main notebook
├── best_model_xgboost.pkl        ← Saved model
├── requirements.txt              ← Libraries
└── README.md                     ← Documentation

## ⚙️ How To Run
1. Clone this repo
2. Install requirements:
3. Open `restaurant_forecasting.ipynb`
4. Run all cells!

## 👨‍💻 Built By
**Rahul Maurya** | Data Science & ML Intern @ Infotact
