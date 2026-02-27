# ================================
# HOUSE PRICE PREDICTION PROJECT
# ================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Connect to MongoDB
# -------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["house_price_db"]
collection = db["house_prices"]

# -------------------------------
# 2. Load CSV into MongoDB (Run once)
# -------------------------------
csv_path = "C:/Users/Ayesha Ismail/Downloads/house_price_prediction_dataset.csv"
data = pd.read_csv(csv_path)

# Insert into MongoDB (only first time)
if collection.count_documents({}) == 0:
    collection.insert_many(data.to_dict("records"))
    print("Data inserted into MongoDB successfully!")
else:
    print("Data already exists in MongoDB.")

# -------------------------------
# 3. Fetch Data from MongoDB
# -------------------------------
mongo_data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(mongo_data)

print("\nDataset Preview:")
print(df.head())

# -------------------------------
# 4. Exploratory Data Analysis (Graphs)
# -------------------------------

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.show()

# Square Feet vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Square_Feet"], y=df["Price"])
plt.title("Square Feet vs Price")
plt.show()

# -------------------------------
# 5. Prepare Data for ML
# -------------------------------
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 7. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 8. Model Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# -------------------------------
# 9. Actual vs Predicted Graph
# -------------------------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

print("\nProject Completed Successfully!")