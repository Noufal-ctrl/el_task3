import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D


# Loading dataset
df = pd.read_csv("/content/Housing.csv")
print(df.head())

# Preprocessing droping the NA
df.dropna(inplace=True)

# Simple Linear Regression
X = df[['area']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# Prediction & Evaluation
y_pred = model.predict(X_test)


# MSE,MAE,R^2
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel('area')
plt.ylabel('price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)



#--------------Multiple Linear Regression--------------#
X = df[['area', 'bedrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("\n[Multiple Regression]")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Features:", X.columns.tolist())

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['area'], X_test['bedrooms'], y_test, color='blue', label='Actual')
ax.scatter(X_test['area'], X_test['bedrooms'], y_pred, color='red', label='Predicted')
ax.set(xlabel='area', ylabel='bedrooms', zlabel='price', title='Multiple Linear Regression (3D)')
ax.legend()
plt.show()
