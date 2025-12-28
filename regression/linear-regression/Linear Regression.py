import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Copy dataset
dataset = df.copy()

# ---- FIX 1: Separate features and target correctly ----
X = dataset.drop("MedHouseVal", axis=1)
y = dataset["MedHouseVal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ---- FIX 2: Scaling ----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
regression = LinearRegression()
regression.fit(X_train, y_train)

# Cross-validation (MSE)
mse = cross_val_score(
    regression,
    X_train,
    y_train,
    scoring="neg_mean_squared_error",
    cv=10
)
print("Mean CV MSE:", np.mean(mse))

# Predictions
reg_pred = regression.predict(X_test)

# Residual distribution
sns.displot(reg_pred - y_test, kind="kde")

# ---- FIX 3: Correct R² order ----
score = r2_score(y_test, reg_pred)
print("R2 Score:", score)
