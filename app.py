"""
Retail Sales Intelligence Dashboard
====================================
Single-file Streamlit app covering:
  Page 1 - Sales Overview Dashboard
  Page 2 - Forecast Explorer
  Page 3 - Anomaly Report
  Page 4 - Product Demand Segments

Run with:
    streamlit run app.py

Expects "train.csv" (Superstore-style sales data) in the same folder,
or upload it via the sidebar file uploader.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


st.sidebar.title("📊 Retail Sales Dashboard")
uploaded = st.sidebar.file_uploader("Upload train.csv (optional)", type=["csv"])

try:
    if uploaded is not None:
        df = load_data(uploaded)
    else:
        df = load_data("train.csv")
except FileNotFoundError:
    st.error(
        "Could not find `train.csv` in the app folder. "
        "Please place it alongside app.py or upload it using the sidebar."
    )
    st.stop()

page = st.sidebar.radio(
    "Navigate",
    [
        "1️⃣ Sales Overview",
        "2️⃣ Forecast Explorer",
        "3️⃣ Anomaly Report",
        "4️⃣ Product Demand Segments",
    ],
)

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ==========================================================================
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ==========================================================================
if page == "1️⃣ Sales Overview":
    st.title("📈 Sales Overview Dashboard")

    # ---- Filters ----
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        regions = st.multiselect(
            "Filter by Region", sorted(df["Region"].unique()),
            default=sorted(df["Region"].unique()),
        )
    with col_f2:
        categories = st.multiselect(
            "Filter by Category", sorted(df["Category"].unique()),
            default=sorted(df["Category"].unique()),
        )

    fdf = df[df["Region"].isin(regions) & df["Category"].isin(categories)]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    # ---- KPIs ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"${fdf['Sales'].sum():,.0f}")
    k2.metric("Total Orders", f"{fdf['Order ID'].nunique():,}")
    k3.metric("Avg Order Value", f"${fdf.groupby('Order ID')['Sales'].sum().mean():,.2f}")
    k4.metric("Years Covered", f"{fdf['Year'].min()}–{fdf['Year'].max()}")

    st.markdown("---")

    # ---- Total sales by year (bar chart) ----
    st.subheader("Total Sales by Year")
    yearly = fdf.groupby("Year")["Sales"].sum().reset_index()
    fig_year = px.bar(
        yearly, x="Year", y="Sales", text_auto=".2s",
        color="Sales", color_continuous_scale="Blues",
    )
    fig_year.update_layout(yaxis_title="Total Sales ($)", showlegend=False)
    st.plotly_chart(fig_year, use_container_width=True)

    # ---- Monthly sales trend line chart ----
    st.subheader("Monthly Sales Trend")
    monthly = (
        fdf.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )
    fig_month = px.line(
        monthly, x="Order Date", y="Sales", markers=True,
    )
    fig_month.update_layout(yaxis_title="Total Sales ($)", xaxis_title="Date")
    st.plotly_chart(fig_month, use_container_width=True)

    # ---- Sales by region & category (interactive) ----
    st.subheader("Sales by Region & Category")
    c1, c2 = st.columns(2)
    with c1:
        reg_cat = fdf.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig_reg = px.bar(
            reg_cat, x="Region", y="Sales", color="Category",
            barmode="group", text_auto=".2s",
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    with c2:
        cat_sales = fdf.groupby("Category")["Sales"].sum().reset_index()
        fig_pie = px.pie(
            cat_sales, names="Category", values="Sales", hole=0.45,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Monthly Seasonality (All Years Combined)")
    seas = fdf.groupby("Month Name")["Sales"].sum().reindex(MONTH_ORDER).reset_index()
    seas.columns = ["Month", "Sales"]
    fig_seas = px.bar(seas, x="Month", y="Sales", color="Sales", color_continuous_scale="Viridis")
    st.plotly_chart(fig_seas, use_container_width=True)


# ==========================================================================
# PAGE 2 — FORECAST EXPLORER
# ==========================================================================
elif page == "2️⃣ Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.caption(
        "Best model: Gradient Boosting Regressor on lag/rolling/calendar "
        "features (per notebook Task 3 & 4 model comparison)."
    )

    dim = st.selectbox("Select dimension to forecast", ["Category", "Region"])
    value = st.selectbox(f"Select {dim}", sorted(df[dim].unique()))

    horizon = st.select_slider(
        "Forecast horizon (months ahead)", options=[1, 2, 3], value=3
    )

    @st.cache_data
    def build_monthly_series(_df, dim, value):
        sub = _df[_df[dim] == value].copy()
        monthly = (
            sub.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
            .sum()
            .to_frame()
        )
        monthly.index.freq = "ME"
        return monthly

    @st.cache_data
    def train_and_forecast(_df, dim, value, horizon):
        monthly = build_monthly_series(_df, dim, value)

        feat = monthly.copy()
        feat["Lag1"] = feat["Sales"].shift(1)
        feat["Lag2"] = feat["Sales"].shift(2)
        feat["Lag3"] = feat["Sales"].shift(3)
        feat["Rolling_Mean"] = feat["Sales"].shift(1).rolling(window=3).mean()
        feat["Month"] = feat.index.month
        feat["Quarter"] = feat.index.quarter
        feat["Season"] = feat["Month"] % 12 // 3 + 1
        feat = feat.dropna()

        feature_cols = ["Lag1", "Lag2", "Lag3", "Rolling_Mean", "Month", "Quarter", "Season"]

        n_test = min(3, max(1, len(feat) // 5))
        if len(feat) <= n_test + 5:
            return None  # not enough data

        X, y = feat[feature_cols], feat["Sales"]
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)

        test_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        # Refit on ALL data, then recursively forecast `horizon` months forward
        model_full = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
        )
        model_full.fit(X, y)

        history = monthly["Sales"].tolist()
        last_date = monthly.index[-1]
        future_dates, future_vals = [], []

        for step in range(horizon):
            next_date = last_date + pd.offsets.MonthEnd(step + 1)
            lag1, lag2, lag3 = history[-1], history[-2], history[-3]
            roll_mean = np.mean(history[-3:])
            row = pd.DataFrame([{
                "Lag1": lag1, "Lag2": lag2, "Lag3": lag3,
                "Rolling_Mean": roll_mean,
                "Month": next_date.month,
                "Quarter": next_date.quarter,
                "Season": next_date.month % 12 // 3 + 1,
            }])[feature_cols]
            pred = model_full.predict(row)[0]
            future_dates.append(next_date)
            future_vals.append(pred)
            history.append(pred)

        forecast_df = pd.DataFrame({"Order Date": future_dates, "Forecast": future_vals})
        return monthly, forecast_df, mae, rmse

    result = train_and_forecast(df, dim, value, horizon)

    if result is None:
        st.warning("Not enough historical data for this selection to build a reliable forecast.")
    else:
        monthly, forecast_df, mae, rmse = result

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly.index, y=monthly["Sales"],
            mode="lines+markers", name="Historical Sales", line=dict(color="royalblue"),
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Order Date"], y=forecast_df["Forecast"],
            mode="lines+markers", name=f"Forecast (+{horizon}mo)",
            line=dict(color="firebrick", dash="dash"),
        ))
        fig.update_layout(
            title=f"Sales Forecast — {value} ({dim})",
            xaxis_title="Date", yaxis_title="Sales ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecasted Values")
        st.dataframe(
            forecast_df.assign(Forecast=lambda d: d["Forecast"].round(2)),
            use_container_width=True,
        )

        st.subheader("Model Performance (Gradient Boosting, held-out test window)")
        m1, m2 = st.columns(2)
        m1.metric("MAE", f"${mae:,.2f}")
        m2.metric("RMSE", f"${rmse:,.2f}")


# ==========================================================================
# PAGE 3 — ANOMALY REPORT
# ==========================================================================
elif page == "3️⃣ Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.caption("Weekly sales anomaly detection using Isolation Forest (Task 5).")

    @st.cache_data
    def detect_anomalies(_df, contamination):
        weekly = _df.set_index("Order Date")["Sales"].resample("W").sum()
        weekly_df = weekly.reset_index()
        weekly_df.columns = ["Order Date", "Sales"]

        iso = IsolationForest(contamination=contamination, random_state=42)
        weekly_df["Anomaly"] = iso.fit_predict(weekly_df[["Sales"]])
        weekly_df["Anomaly"] = weekly_df["Anomaly"].map({1: "Normal", -1: "Anomaly"})
        return weekly_df

    contamination = st.slider(
        "Sensitivity (expected anomaly proportion)", 0.01, 0.15, 0.05, 0.01
    )

    weekly_df = detect_anomalies(df, contamination)
    anomalies = weekly_df[weekly_df["Anomaly"] == "Anomaly"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_df["Order Date"], y=weekly_df["Sales"],
        mode="lines", name="Weekly Sales", line=dict(color="steelblue"),
    ))
    fig.add_trace(go.Scatter(
        x=anomalies["Order Date"], y=anomalies["Sales"],
        mode="markers", name="Anomaly",
        marker=dict(color="red", size=10, symbol="x"),
    ))
    fig.update_layout(
        title="Isolation Forest — Weekly Sales Anomalies",
        xaxis_title="Date", yaxis_title="Sales ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    k1, k2 = st.columns(2)
    k1.metric("Total Weeks", len(weekly_df))
    k2.metric("Anomalies Detected", len(anomalies))

    st.subheader("Detected Anomaly Dates")
    st.dataframe(
        anomalies[["Order Date", "Sales"]]
        .sort_values("Order Date")
        .assign(Sales=lambda d: d["Sales"].round(2))
        .reset_index(drop=True),
        use_container_width=True,
    )


# ==========================================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ==========================================================================
elif page == "4️⃣ Product Demand Segments":
    st.title("🧩 Product Demand Segments")
    st.caption("K-Means clustering of sub-categories by volume, AOV, volatility & growth (Task 6).")

    n_clusters = st.slider("Number of clusters (K)", 2, 6, 4)

    @st.cache_data
    def build_cluster_features(_df):
        total_sales = _df.groupby("Sub-Category")["Sales"].sum()
        unique_orders = _df.groupby("Sub-Category")["Order ID"].nunique()
        aov = total_sales / unique_orders

        monthly_subcat = (
            _df.groupby(["Sub-Category", "Year", "Month"])["Sales"].sum().reset_index()
        )
        volatility = monthly_subcat.groupby("Sub-Category")["Sales"].std().fillna(0)

        yearly_sales = (
            _df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack().fillna(0)
        )
        years_sorted = sorted(yearly_sales.columns)
        latest, prev = years_sorted[-1], years_sorted[-2] if len(years_sorted) > 1 else years_sorted[-1]
        growth = ((yearly_sales[latest] - yearly_sales[prev]) / yearly_sales[prev].replace(0, np.nan)) * 100
        growth = growth.fillna(0)

        features = pd.DataFrame({
            "Total Sales": total_sales,
            "Avg Order Value": aov,
            "Volatility": volatility,
            "YoY Growth %": growth,
        }).fillna(0)

        return features

    @st.cache_data
    def run_kmeans(features, k):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        result = features.copy()
        result["Cluster"] = labels.astype(str)
        result["PC1"] = coords[:, 0]
        result["PC2"] = coords[:, 1]
        return result

    features = build_cluster_features(df)
    clustered = run_kmeans(features, n_clusters)

    fig = px.scatter(
        clustered.reset_index(), x="PC1", y="PC2", color="Cluster",
        text="Sub-Category", size="Total Sales", size_max=40,
        hover_data=["Total Sales", "Avg Order Value", "Volatility", "YoY Growth %"],
        title="Sub-Category Clusters (PCA Projection)",
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Categories by Demand Cluster")
    display_tbl = (
        clustered.reset_index()
        .rename(columns={"index": "Sub-Category"})
        .sort_values(["Cluster", "Total Sales"], ascending=[True, False])
    )
    display_tbl["Total Sales"] = display_tbl["Total Sales"].round(2)
    display_tbl["Avg Order Value"] = display_tbl["Avg Order Value"].round(2)
    display_tbl["Volatility"] = display_tbl["Volatility"].round(2)
    display_tbl["YoY Growth %"] = display_tbl["YoY Growth %"].round(2)

    st.dataframe(
        display_tbl[["Cluster", "Sub-Category", "Total Sales", "Avg Order Value",
                      "Volatility", "YoY Growth %"]].reset_index(drop=True),
        use_container_width=True,
    )
