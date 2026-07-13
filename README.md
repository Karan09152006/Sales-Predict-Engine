# 📈 Retail Sales Intelligence Dashboard & Sales Prediction Engine

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?style=for-the-badge&logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-f7931e?style=for-the-badge&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visualization-blueviolet?style=for-the-badge&logo=plotly)

</p>

---

# 📌 Project Overview

The **Retail Sales Intelligence Dashboard & Sales Prediction Engine** is an end-to-end Data Science project developed to analyze historical retail sales data, forecast future sales, detect anomalies, and segment products based on demand patterns.

The project combines **Business Intelligence**, **Machine Learning**, and **Interactive Dashboarding** into a single application that enables businesses to monitor sales performance, identify unusual sales trends, forecast future demand, and make data-driven inventory decisions.

This project demonstrates practical applications of:

- 📊 Exploratory Data Analysis (EDA)
- 📈 Time Series Analysis
- 🤖 Machine Learning Forecasting
- 🚨 Anomaly Detection
- 📦 Product Demand Segmentation
- 📉 Business Intelligence Dashboard
- 📊 Interactive Data Visualization

---

# 🎯 Project Objectives

The primary objectives of this project are:

- Analyze historical retail sales performance
- Identify seasonal sales trends
- Forecast future monthly sales
- Detect unusual sales spikes and drops
- Segment products according to demand patterns
- Build an interactive dashboard for business users
- Generate business recommendations using data-driven insights

---

# 🚀 Key Features

### 📊 Sales Dashboard

- Total Sales KPI
- Total Profit KPI
- Total Orders
- Monthly Sales Trend
- Regional Sales Analysis
- Category-wise Sales
- Sub-category Performance
- Interactive Filters

---

### 🔮 Sales Forecasting

- Monthly Sales Forecast
- Machine Learning Prediction
- Trend Analysis
- Future Sales Projection
- Forecast Confidence Visualization

---

### 🚨 Anomaly Detection

- Weekly Sales Analysis
- Isolation Forest Algorithm
- Sales Spike Detection
- Sales Drop Detection
- Automatic Outlier Identification

---

### 📦 Product Demand Segmentation

- Customer/Product Segmentation
- K-Means Clustering
- PCA Visualization
- Demand Classification
- Inventory Planning Support

---

# 📂 Dataset Description

The project uses the **Superstore Sales Dataset**, which contains historical retail transactions.

## Dataset Features

| Feature | Description |
|----------|-------------|
| Order ID | Unique order identifier |
| Order Date | Date of purchase |
| Ship Date | Shipping date |
| Customer Name | Customer information |
| Segment | Customer segment |
| Region | Sales region |
| Category | Product category |
| Sub-Category | Product sub-category |
| Product Name | Product details |
| Sales | Total sales amount |
| Quantity | Number of products sold |
| Discount | Discount offered |
| Profit | Profit generated |

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Data Analysis

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Plotly

## Machine Learning

- Scikit-Learn
- XGBoost

## Time Series Analysis

- Statsmodels

## Dashboard

- Streamlit

---

# 📂 Project Workflow

```text
                Retail Sales Dataset
                        │
                        ▼
               Data Cleaning & Preprocessing
                        │
                        ▼
             Exploratory Data Analysis (EDA)
                        │
                        ▼
             Time Series Decomposition
                        │
                        ▼
              Machine Learning Forecasting
                        │
                        ▼
               Anomaly Detection
                        │
                        ▼
          Product Demand Segmentation
                        │
                        ▼
          Interactive Streamlit Dashboard
                        │
                        ▼
          Business Insights & Recommendations
```

---

# 📊 Exploratory Data Analysis

The project performs detailed EDA to understand the retail sales data.

### Analysis Performed

- Monthly Sales Trend
- Regional Sales Analysis
- Category-wise Sales
- Sub-category Performance
- Shipping Time Analysis
- Discount Analysis
- Profit Analysis
- Seasonal Trend Analysis
- Correlation Analysis

---

# 📈 Graphs & Visualizations

The project contains multiple visualizations to better understand business performance.

## 📊 Monthly Sales Trend

```markdown
![Monthly Sales](https://github.com/Karan09152006/Sales-Predict-Engine/blob/95374b40f1ecfbe477affe4619c37bebf42701de/charts/Monthly%20Sales.png)
```

---

## 🌍 Regional Sales Analysis

```markdown
![Region wise sales growth](https://github.com/Karan09152006/Sales-Predict-Engine/blob/95374b40f1ecfbe477affe4619c37bebf42701de/charts/Region-wise%20Sales%20Growth.png)
```

---

## 🛒 Category-wise Sales

```markdown
![Category Sales](https://github.com/Karan09152006/Sales-Predict-Engine/blob/95374b40f1ecfbe477affe4619c37bebf42701de/charts/Sales%20by%20Category.png)
```

---

## 📈 Time Series Decomposition

```markdown
![Time Series](https://github.com/Karan09152006/Sales-Predict-Engine/blob/95374b40f1ecfbe477affe4619c37bebf42701de/charts/time_series_decomposition.png)
```

---

## 📦 Product Demand Segmentation

```markdown
![Clustering](https://github.com/Karan09152006/Sales-Predict-Engine/blob/95374b40f1ecfbe477affe4619c37bebf42701de/charts/product_clusters_chart.png)
```

---

# 🤖 Machine Learning Models

The project implements multiple Machine Learning and Time Series techniques for forecasting, anomaly detection, and product segmentation.

## Models Used

| Model | Purpose |
|--------|---------|
| Linear Regression | Baseline Forecasting |
| Random Forest Regressor | Sales Prediction |
| XGBoost Regressor | Advanced Sales Forecasting |
| Isolation Forest | Weekly Sales Anomaly Detection |
| K-Means Clustering | Product Demand Segmentation |
| PCA | Cluster Visualization |
| Seasonal Decomposition | Trend & Seasonality Analysis |

---

# 🔮 Sales Forecasting

The forecasting module predicts future monthly sales using historical sales trends.

## Forecasting Pipeline

```text
Historical Sales
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train-Test Split
        │
        ▼
Machine Learning Model
        │
        ▼
Sales Prediction
        │
        ▼
Future Sales Forecast
```

### Forecast Features

- Monthly Sales Forecast
- Future Trend Analysis
- Growth Prediction
- Seasonal Pattern Detection
- Interactive Forecast Visualization

---

# 🚨 Anomaly Detection

The project detects unusual sales patterns using **Isolation Forest**.

### Detects

- Sudden Sales Spike
- Unexpected Sales Drop
- Seasonal Outliers
- Inventory Issues
- Promotional Events

### Business Benefits

- Early identification of unusual sales behavior
- Detect promotional impacts
- Improve inventory planning
- Identify potential operational issues

---

# 📦 Product Demand Segmentation

Products are grouped into demand-based clusters using **K-Means Clustering**.

## Features Used

- Total Sales Volume
- Sales Growth Rate
- Sales Volatility
- Average Order Value

### Demand Segments

| Cluster | Business Meaning |
|----------|------------------|
| High Volume, Stable Demand | High-selling products with consistent demand |
| Growing Demand | Products showing increasing sales trends |
| Low Volume, High Volatility | Products with inconsistent demand |
| Declining Demand | Products with decreasing sales trends |

---

# 📊 Dashboard Features

The Streamlit dashboard provides an interactive interface for business users.

## Dashboard Modules

### 📈 Sales Overview

- Total Sales KPI
- Total Profit KPI
- Total Orders
- Average Order Value
- Sales Trend

---

### 🌍 Regional Analysis

- Region-wise Sales
- State-wise Performance
- Profit Distribution
- Regional Comparison

---

### 🛒 Product Analysis

- Category Performance
- Sub-category Analysis
- Top Products
- Product-wise Sales

---

### 🔮 Forecast Explorer

- Monthly Forecast
- Historical vs Predicted Sales
- Future Trend
- Forecast Comparison

---

### 🚨 Anomaly Detection

- Weekly Sales Trend
- Highlighted Anomalies
- Outlier Analysis

---

### 📦 Product Segmentation

- K-Means Clusters
- PCA Visualization
- Cluster Summary
- Demand Classification

---

## 🌐 Live Demo

Experience the interactive dashboard here:

[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-sales-dashbo-g8tz6mst5chm7vjvpbh4hv.streamlit.app/)

### Features Available

- 📊 Interactive Sales Dashboard
- 📈 Sales Forecasting
- 🚨 Anomaly Detection
- 📦 Product Demand Segmentation
- 📉 Business Insights
- 🔍 Interactive Filters

  
---

# 📷 Dashboard Preview

Place your dashboard screenshots inside the **images/** folder.

## Main Dashboard

```markdown
![Dashboard](images/dashboard.png)
```

---

## Forecast Dashboard

```markdown
![Forecast Dashboard](images/forecast_dashboard.png)
```

---

## Anomaly Detection Dashboard

```markdown
![Anomaly Dashboard](images/anomaly_dashboard.png)
```

---

## Product Segmentation Dashboard

```markdown
![Segmentation Dashboard](images/segmentation_dashboard.png)
```

---


# 💡 Business Insights

The analysis generated several valuable business insights.

### Key Findings

- Sales exhibit strong seasonal trends with significant spikes during festive months.
- Technology products generate the highest revenue among all product categories.
- The West region consistently records the highest sales.
- Some weeks experience unusual sales spikes due to promotional campaigns.
- High-demand products should be prioritized for inventory planning.
- Certain low-performing products may require promotional strategies or inventory optimization.

---

# 📈 Business Recommendations

Based on the analysis, the following recommendations are proposed:

- Increase inventory for high-demand products.
- Focus marketing efforts during high-performing seasons.
- Monitor anomalies to identify unusual business events.
- Optimize inventory for low-demand products.
- Expand successful product categories into high-growth regions.
- Use predictive forecasting for demand planning and procurement.

---

# 📊 Results

The developed system successfully provides:

- Accurate sales forecasting
- Weekly anomaly detection
- Product demand segmentation
- Interactive dashboard for decision-making
- Business intelligence insights
- Visual trend analysis

The project demonstrates how Machine Learning can support retail organizations in improving sales planning, inventory management, and strategic business decisions.

---

# 📁 Project Structure

```text
Retail-Sales-Intelligence-Dashboard/
│
├── app.py
├── analysis.ipynb
├── README.md
├── requirements.txt
├── train.csv
│
├── images/
│   ├── dashboard.png
│   ├── monthly_sales.png
│   ├── regional_sales.png
│   ├── category_sales.png
│   ├── subcategory_sales.png
│   ├── profit_analysis.png
│   ├── time_series.png
│   ├── forecast.png
│   ├── forecast_dashboard.png
│   ├── anomaly_detection.png
│   ├── anomaly_dashboard.png
│   ├── clustering.png
│   └── segmentation_dashboard.png
│
└── models/
    ├── forecasting_model.pkl
    └── clustering_model.pkl
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Retail-Sales-Intelligence-Dashboard.git
```

---

## Navigate to Project Folder

```bash
cd Retail-Sales-Intelligence-Dashboard
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 📦 Requirements

```text
pandas
numpy
matplotlib
plotly
streamlit
scikit-learn
xgboost
statsmodels
joblib
openpyxl
```

---

# 🚀 Future Improvements

Future enhancements for this project include:

- Deep Learning-based sales forecasting (LSTM)
- Real-time sales prediction
- Automated model retraining
- Cloud deployment (AWS/Azure)
- Power BI integration
- Customer segmentation
- Recommendation system
- Interactive business reporting
- Sales alert notifications
- Mobile dashboard support

---

# 👨‍💻 Author

## Dharmendra Chhapola

**Artificial Intelligence & Data Science Student**

📍 Jaipur, Rajasthan, India

📧 dharmendrachhapola@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/dharmendra-singh-chhapola-14907628a/

💻 GitHub: https://github.com/yourusername

---

# 📜 License

This project is licensed under the **MIT License**.

---

# ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork the repository

📢 Share it with others

Your support is greatly appreciated!

---
