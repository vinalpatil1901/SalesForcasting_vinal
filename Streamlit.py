# ==============================
# STREAMLIT DASHBOARD
# ==============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("Sales Forecasting Dashboard")

# Load Files
sales = pd.read_csv("cleaned_sales.csv")
forecast = pd.read_csv("forecast.csv")
segments = pd.read_csv("product_segments.csv")
anomalies = pd.read_csv("anomalies.csv")

# Sidebar Filter
region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + list(sales["Region"].unique())
)

if region != "All":
    sales = sales[sales["Region"] == region]

# KPI Cards
c1, c2, c3 = st.columns(3)

c1.metric("Total Sales", f"${sales['Sales'].sum():,.0f}")
c2.metric("Orders", sales["Order ID"].nunique())
c3.metric("Customers", sales["Customer ID"].nunique())

# Sales by Category
st.subheader("Sales by Category")
cat = sales.groupby("Category")["Sales"].sum().reset_index()

plt.figure(figsize=(6,4))
plt.bar(cat["Category"], cat["Sales"])
plt.title("Sales by Category")
st.pyplot(plt)

# Sales by Region
st.subheader("Sales by Region")
reg = sales.groupby("Region")["Sales"].sum().reset_index()

plt.figure(figsize=(6,6))
plt.pie(reg["Sales"], labels=reg["Region"], autopct="%1.1f%%")
plt.title("Sales by Region")
st.pyplot(plt)

# Monthly Sales Trend
st.subheader("Monthly Sales Trend")
month = sales.groupby("Month")["Sales"].sum().reset_index()

plt.figure(figsize=(8,4))
plt.plot(month["Month"], month["Sales"], marker="o")
plt.title("Monthly Sales Trend")
st.pyplot(plt)

# Sales Forecast
st.subheader("Sales Forecast")
forecast["Date"] = pd.to_datetime(forecast["Date"])

plt.figure(figsize=(8,4))
plt.plot(forecast["Date"], forecast["Actual"], label="Actual")
plt.plot(forecast["Date"], forecast["Forecast"], label="Forecast")
plt.legend()
plt.title("Sales Forecast")
st.pyplot(plt)

# Product Segmentation
st.subheader("Product Segmentation")

plt.figure(figsize=(7,5))
plt.scatter(segments["Sales"], segments["Shipping Days"], c=segments["Segment"])
plt.title("Product Segmentation")
st.pyplot(plt)

# Anomaly Detection
st.subheader("Anomaly Detection")

plt.figure(figsize=(7,5))
plt.scatter(anomalies["Shipping Days"], anomalies["Sales"], c=anomalies["Anomaly"])
plt.title("Anomaly Detection")
st.pyplot(plt)

# Raw Data
if st.checkbox("Show Dataset"):
    st.dataframe(sales)
