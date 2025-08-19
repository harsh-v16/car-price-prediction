# =================================
# 1. Importing Required Libraries
# =================================
# pandas → for data manipulation and analysis
# numpy  → for mathematical operations
# matplotlib & seaborn → for data visualization
# scikit-learn → for preprocessing, model building, and evaluation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==========================
# 2. Load Dataset
# ==========================
# Reading the dataset (CSV file) into a pandas DataFrame
# Make sure the dataset 'CarPrice_Assignment.csv' is in the same folder

train_data = pd.read_csv('CarPrice_Assignment.csv')

# Display first 5 rows to verify successful loading
print(train_data.head())

# --- PART 2: EXPLORATORY DATA ANALYSIS (EDA) & PREPROCESSING ---
# The goal here is to fix the skewness in the target variable (SalePrice),
# understand feature relationships, handle missing values, and prepare
# categorical features for the model.

# 2a. Skewness Analysis and Correction
# First, we visually inspect the distribution of SalePrice using a histogram
# and a Q-Q plot.

fig,axes = plt.subplots(1, 2, figsize = (15, 5))
sns.histplot(train_data['price'], kde=True, ax=axes[0])
axes[0].set_title('Histogram of Price')
stats.probplot(train_data['price'], plot=axes[1])
axes[1].set_title('Q-Q Plot of Price')
plt.show()

# We then perform a mathematical check for skewness. A value close to 0
# indicates a symmetrical distribution. A value > 1 indicates high positive skew

print(f"Skewness : {train_data['price'].skew()}")

# To correct the skewness, we apply a log transformation. This helps the model
# make more reliable predictions as many models assume a normal distribution.

train_data['price_log'] = np.log1p(train_data['price'])
fig,axes = plt.subplots(1, 2, figsize = (15, 5))
sns.histplot(train_data['price_log'], kde=True, ax=axes[0])
axes[0].set_title('Histogram of Log-Transformed price')
stats.probplot(train_data['price_log'], plot=axes[1])
axes[1].set_title('Q-Q Plot of Log-Transformed price')
plt.show()

# We check the new skewness value to confirm our transformation was successful.

print(f"New Skewness : {train_data['price_log'].skew()}")

# 2b. Correlation Analysis for Feature Selection
# We create a correlation heatmap to understand the relationships between
# features and our target variable, 'SalePrice_log'. This helps us select
# the most powerful and relevant features for a simple baseline model and
# understand potential multicollinearity (features that are highly
# correlated with each other)

corrmat = train_data.corr(numeric_only=True)
k = 10
top_10_cols = corrmat.nlargest(k, 'price_log')['price_log'].index
top_10_corrmat = train_data[top_10_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(top_10_corrmat, annot=True, cmap="YlGnBu", fmt='.2f')
plt.title('Top 10 Feature Correlations with price_log')
plt.show()

# 2c. Handling Missing Values (no missing values found)

missing_vals = train_data.isnull().sum()
cols_with_missing = missing_vals[missing_vals > 0].index.tolist()

numerical_cols_missing = train_data[cols_with_missing].select_dtypes(include=np.number).columns.tolist()
categorical_cols_missing = train_data[cols_with_missing].select_dtypes(include='object').columns.tolist()

print("Numeric empty spots")
print(numerical_cols_missing)
print("Categorical empty spots")
print(categorical_cols_missing)

# 2d. One-Hot Encoding

categorical_cols = [cname for cname in train_data.columns if train_data[cname].dtype == 'object']
train_data = pd.get_dummies(train_data, columns=categorical_cols)

# --- PART 3: MODELING & EVALUATION ---
# Now we will build two models:
# 1. A simple Linear Regression model as a baseline.
# 2. A powerful XGBoost model to achieve high performance.
# We will compare their R-squared and RMSE scores to choose a winner.

y = train_data['price_log']

selected_features = ['curbweight', 'enginesize', 'horsepower', 'boreratio']
X_simple = train_data[selected_features]

X_simple_train, X_simple_val, y_simple_train, y_simple_val = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# 3a. Baseline Model: Linear Regression

simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_simple_train)
predictions = simple_model.predict(X_simple_val)

r2 = r2_score(y_simple_val, predictions)
print(f"R-squared (R2 Score): {r2:.4f}")

mse = mean_squared_error(y_simple_val, predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

X_complex = train_data.drop(['price', 'price_log'], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_complex, y, test_size=0.2, random_state=42)

# 3b. High-Performance Model: XGBoost

complex_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)      
complex_model.fit(X_train, y_train) 
complex_predictions = complex_model.predict(X_val)  

r2 = r2_score(y_val, complex_predictions)
print(f"R-squared (R2 Score): {r2:.4f}")

mse = mean_squared_error(y_val, complex_predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# --- PART 4: FINAL MODEL PREPARATION & PREDICTION ---
# After proving that XGBoost is the superior model, our final step is to
# prepare it for a "production" environment. We re-train the model on 100%
# of the available data to make it as accurate as possible. We then use this
# final model to generate a sample prediction file, simulating how it
# would be used on new data in a real business scenario.

final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)      
final_model.fit(X_complex, y) 
predictions_log = final_model.predict(X_complex)
predictions_actual = np.expm1(predictions_log)

results_df = pd.DataFrame({
    'Car_ID': train_data.index,
    'Predicted_Price': predictions_actual
})

results_df.to_csv('car_price_predictions.csv', index=False)

print("\n--- SUCCESS! ---")
print("Submission file 'submission.csv' has been created.")






             
     