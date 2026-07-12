# ==============================
# Data Cleaning + EDA
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"C:\college\PROJECTS\SalesForecasting_Vinal\archive1\train.csv")

# -----------------------------
# Data Cleaning
# -----------------------------

# Missing Values
df["Postal Code"] = df["Postal Code"].fillna(df["Postal Code"].median())

# Remove Duplicates
df = df.drop_duplicates()

# Convert Date Columns
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

# Feature Engineering
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Quarter"] = df["Order Date"].dt.quarter
df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

# Save Clean Data
df.to_csv("cleaned_sales.csv", index=False)

print(df.head())
print(df.info())
print(df.describe())

# ==============================
# Exploratory Data Analysis
# ==============================

# Sales by Category
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="Category", y="Sales", estimator=sum, errorbar=None)
plt.title("Sales by Category")
plt.savefig(r"C:\college\PROJECTS\SalesForecasting_Vinal\charts\sales_by_category.png")
plt.show()

# Sales by Region
plt.figure(figsize=(6,4))
sns.barplot(data=df, x="Region", y="Sales", estimator=sum, errorbar=None)
plt.title("Sales by Region")
plt.savefig(r"C:\college\PROJECTS\SalesForecasting_Vinal\charts\sales_by_region.png")
plt.show()

# Monthly Sales
monthly = df.groupby("Month")["Sales"].sum().reset_index()

plt.figure(figsize=(7,4))
sns.lineplot(data=monthly, x="Month", y="Sales", marker="o")
plt.title("Monthly Sales Trend")
plt.savefig(r"C:\college\PROJECTS\SalesForecasting_Vinal\charts\monthly_sales_trend.png")
plt.show()

# Top 10 Products
top = df.groupby("Product Name")["Sales"].sum().nlargest(10).reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=top, x="Sales", y="Product Name")
plt.title("Top 10 Products")
plt.savefig(r"C:\college\PROJECTS\SalesForecasting_Vinal\charts\top_10_products.png")   
plt.show()

# Correlation
plt.figure(figsize=(7,5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.savefig(r"C:\college\PROJECTS\SalesForecasting_Vinal\charts\correlation_heatmap.png")
plt.show()
