import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Rest of the code remains the same...

# Generate sample data
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Create Polynomial Regression model
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

polynomial_model = LinearRegression()
polynomial_model.fit(X_train_poly, y_train)

# Evaluate models on test set
y_pred_linear = linear_model.predict(X_test)
y_pred_poly = polynomial_model.predict(X_test_poly)

mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_poly = mean_squared_error(y_test, y_pred_poly)

print("Mean Squared Error (Linear Regression):", mse_linear)
print("Mean Squared Error (Polynomial Regression):", mse_poly)

# Plot the results
plt.scatter(X_test, y_test, color='b', label='Test Data')
plt.plot(X_test, y_pred_linear, color='r', label='Linear Regression')
plt.plot(X_test, y_pred_poly, color='g', label='Polynomial Regression')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear vs. Polynomial Regression')
plt.show()
