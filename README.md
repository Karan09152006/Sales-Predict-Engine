# 📊 Retail Sales Forecasting & Analytics

Time series forecasting, anomaly detection, and product demand segmentation on retail sales data — combining classical statistics, machine learning, and clustering to help businesses plan inventory and reduce stockouts.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 Problem Statement

Retail businesses generate large volumes of sales data every day, but identifying trends, seasonal demand, and future performance from that data is hard. Without accurate forecasting, companies face overstocking, stockouts, inefficient inventory management, and poor planning.

This project analyzes historical retail sales data, understands customer purchasing patterns, and builds forecasting, anomaly detection, and clustering models to help businesses make data-driven inventory and supply-chain decisions.

## 🎯 Key Objectives

- Perform deep Exploratory Data Analysis (EDA) on retail sales
- Analyze time series components (trend, seasonality, residuals)
- Forecast future sales using three different modeling approaches
- Detect sales anomalies using statistical and ML methods
- Segment products by demand behavior using clustering

---

## 🗂️ Project Structure

```
.
├── analysis.ipynb          # Main analysis notebook (all tasks)
├── train.csv                # Raw retail sales dataset
├── images/                  # Exported charts used in this README
└── README.md
```

---

## 🧠 Workflow

### Task 1 — Data Loading, Merging & Deep Exploration
Data cleaning, weekly/monthly aggregation, and exploratory questions around revenue by category, regional growth, shipping time, and seasonality.

**Technology** generates the highest total sales, and the **West region** shows the most consistent growth over 4 years.

| Revenue by Category | Sales by Region | Monthly Seasonality |
|---|---|---|
| ![Revenue by category](images/revenue_by_category.png) | ![Sales by region](images/sales_by_region.png) | ![Monthly seasonality](images/monthly_seasonality.png) |

---

### Task 2 — Time Series Analysis & Decomposition
Overall monthly sales trend, seasonal decomposition (trend/seasonality/residual), an ADF stationarity test, and differencing where needed.

- Steady long-term upward growth across all 4 years
- Strong, predictable seasonality — dip in Q1, spike in Q4 (holiday demand)
- Highest variance in November–December, driven by year-end promotions

| Overall Monthly Trend | Seasonal Decomposition | Differencing / Stationarity |
|---|---|---|
| ![Overall monthly trend](images/overall_monthly_trend.png) | ![Seasonal decomposition](images/seasonal_decomposition.png) | ![Differencing](images/differencing_stationarity.png) |

---

### Task 3 — Sales Forecasting (3 Models)
Three forecasting approaches were trained on historical monthly sales and evaluated on a 3-month holdout window:

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| SARIMA(1,1,1)(1,1,1)₁₂ | 6037.92 | 6039.62 | 35.13 |
| Facebook Prophet | 6165.58 | 6549.34 | 36.06 |
| **Gradient Boosting (ML)** | **3270.99** | **3397.29** | **19.03** |

The **Gradient Boosting model** (lag features + rolling mean + calendar features) clearly outperformed the classical statistical models on this dataset.

| SARIMA Forecast | Prophet Seasonality | XGBoost / GB Forecast |
|---|---|---|
| ![SARIMA forecast](images/sarima_forecast.png) | ![Prophet yearly seasonality](images/prophet_yearly_seasonality.png) | ![Gradient boosting forecast](images/xgboost_forecast.png) |

---

### Task 4 — Category & Region Level Forecasting
Forecasts broken down by product category and region. The **West** and **East** regions show the strongest upcoming growth, led by the **Technology** category.

![Category and region forecast](images/category_region_forecast.png)

---

### Task 5 — Anomaly Detection
Weekly sales anomalies were detected two ways and cross-validated:

- **Isolation Forest** — isolates unusual observations without relying on fixed statistical thresholds
- **Rolling Z-Score** — flags weeks that deviate significantly from a rolling mean/std

Detected spikes align with real-world events like Black Friday / Cyber Monday promotions.

| Isolation Forest Anomalies | Z-Score Anomalies |
|---|---|
| ![Isolation Forest anomalies](images/isolation_forest_anomalies.png) | ![Z-score anomalies](images/zscore_anomalies.png) |

---

### Task 6 — Product Demand Segmentation (Clustering)
Sub-categories were clustered with K-Means (k=4, chosen via the elbow method) on total sales and order-volume features, then projected to 2D with PCA.

| Elbow Method | Product Clusters (PCA) |
|---|---|
| ![Elbow method](images/elbow_method.png) | ![Product clustering PCA](images/product_clustering_pca.png) |

**Segments & recommended inventory strategy:**

| Segment | Profile | Strategy |
|---|---|---|
| High Volume, Stable | Consistent, high-revenue "cash cow" products | Automated replenishment, strict safety stock, bulk supplier contracts |
| Growing Demand | Strong YoY upward momentum | Scale up forecasting, marketing spend, and warehouse space |
| Low Volume, High Volatility | Rare, erratic bulk/B2B purchases | Drop-shipping / on-demand fulfillment, avoid dead stock |
| Declining Demand | Negative YoY growth | Halt reordering, liquidate via discounts/bundling, reallocate space |

---

## 🛠️ Tech Stack

- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Time Series:** statsmodels (SARIMA, seasonal decomposition, ADF test), Facebook Prophet
- **Machine Learning:** scikit-learn (Gradient Boosting, Isolation Forest, KMeans, PCA)

## 🚀 Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

### requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
prophet
```

---

## 📈 Key Takeaways

- Gradient Boosting outperformed SARIMA and Prophet for monthly sales forecasting (MAPE ~19% vs ~35–36%)
- Sales are strongly seasonal, with a reliable Q4 holiday spike and Q1 slump
- November/December carry the highest anomaly risk due to promotional volatility
- Product sub-categories cluster into four clear demand profiles, each warranting a different inventory strategy

## 📄 License

This project is licensed under the MIT License.
