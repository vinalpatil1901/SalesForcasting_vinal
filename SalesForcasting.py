# ==============================
# Forecasting + ML
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ------------------------------
# Load Data
# ------------------------------

df = pd.read_csv("cleaned_sales.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])

# ==============================
# SALES FORECASTING
# ==============================
df["Order Date"] = pd.to_datetime(df["Order Date"])

sales = df.set_index("Order Date")["Sales"].resample("ME").sum()

train = sales[:-12]
test = sales[-12:]

model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
result = model.fit()

forecast = result.forecast(12)

forecast_df = pd.DataFrame({
    "Date": test.index,
    "Actual": test.values,
    "Forecast": forecast.values
})

forecast_df.to_csv("forecast.csv", index=False)

plt.figure(figsize=(10,5))
plt.plot(train, label="Train")
plt.plot(test, label="Actual")
plt.plot(forecast, label="Forecast")
plt.title("Sales Forecast")
plt.legend()
plt.savefig(r"C:\college\PROJECTS\SalesForecasting_Vinal\charts\sales_forecast.png")
plt.show()

# ==============================
# MODEL EVALUATION
# ==============================

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("MAE :", round(mae,2))
print("RMSE :", round(rmse,2))

# ==============================
# ANOMALY DETECTION
# ==============================

iso = IsolationForest(contamination=0.02, random_state=42)

df["Anomaly"] = iso.fit_predict(df[["Sales"]])

df.to_csv("anomalies.csv", index=False)

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Shipping Days", y="Sales", hue="Anomaly")
plt.title("Anomaly Detection")
plt.savefig(r"C:\college\PROJECTS\SalesForecasting_Vinal\charts\anomaly_detection.png")
plt.show()

# ==============================
# PRODUCT SEGMENTATION
# ==============================

kmeans = KMeans(n_clusters=3, random_state=42)

df["Segment"] = kmeans.fit_predict(df[["Sales","Shipping Days"]])

df.to_csv("product_segments.csv", index=False)

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Sales", y="Shipping Days", hue="Segment")
plt.title("Product Segmentation")
plt.savefig(r"C:\college\PROJECTS\SalesForecasting_Vinal\charts\product_segmentation.png")
plt.show()
