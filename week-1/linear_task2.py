import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("house_rent_dataset.csv")

# dropping unnecessary coloumns
df = df.drop(columns=["Posted On", "Point of Contact"])

# cleaning floor coloumn by extracting number and ground/lower

def convert_floor(floor_str):
    floor_str = str(floor_str).lower()
    if "ground" in floor_str:
        return 0
    elif "basement" in floor_str or "lower" in floor_str:
        return -1
    else:
        try:
            return int(floor_str.split()[0])
        except:
            return 0

df["Floor"] = df["Floor"].apply(convert_floor)

# dropping null coloumns 

df = df.dropna()

# binary encoding the coloumns and remove the first one to reduce the computational time

categorical_cols = ["Area Type", "Area Locality", "City", "Furnishing Status", "Tenant Preferred"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# extracting rent date for training and testing 
X = df_encoded.drop("Rent", axis=1)
y = df_encoded["Rent"]

# choosing random 80% data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# training our model
model = LinearRegression()
model.fit(X_train, y_train)

# predicting our model
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# print model coefficients
print("Model Coefficients (Weights):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# printing intercept and mse
print(f"\nIntercept: {model.intercept_:.4f}")
print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}\n")

# print actual vs predicted rents 
print("Actual Rent vs Predicted Rent on Test Set:")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {int(actual):,}  Predicted: {int(round(pred)):,}")


