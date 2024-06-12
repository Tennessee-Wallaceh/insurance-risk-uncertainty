# Trying point estimation on the claim amoount using standard xgboost

import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm


# In previous script I have experimented with XGboost
# In this script I will try to use XGboost with Conformal Prediction

# Load data
data = pd.read_csv('merged_data.csv')
# remove first column
data = data.drop(columns=['Unnamed: 0'])
# remove all rows with ClaimNb equal to zero
data = data[data['ClaimNb'] != 0]
# Only keep the values less than the 95% quantile
data = data[data['ClaimAmount'] < data['ClaimAmount'].quantile(0.95)]

categorical_columns = ['VehPower', 'VehAge', 'VehBrand', 'VehGas', 'Area', 'Region']
for col in categorical_columns:
    data[col] = data[col].astype('category')

X = data.drop(columns=['ClaimAmount'])
y = data['ClaimAmount']

# Split data into train+calibration and validation sets
X_train_cal, X_val, y_train_cal, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train+calibration set into training and calibration sets
X_train, X_cal, y_train, y_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.5, 'alpha': 10
}
xgb_reg = xgb.XGBRegressor(objective='reg:gamma',**params, enable_categorical=True)
best_xgb_reg = xgb.XGBRegressor(**params, enable_categorical=True)
best_xgb_reg.fit(X_train, y_train)



# compute the score function
# Make predictions on the calibration set
cal_preds = best_xgb_reg.predict(X_cal)

# Calculate the absolute errors (for regression)
cal_errors = np.abs(cal_preds - y_cal)


# get the quantile
alpha = 0.1
quantile = np.quantile(cal_errors, 1 - alpha)

# Make predictions on the validation set
val_preds = best_xgb_reg.predict(X_val)

# Create prediction intervals
prediction_intervals = np.array([
    (pred - quantile, pred + quantile) for pred in val_preds
])

# Calculate coverage and average length of the intervals
coverage = np.mean((y_val >= prediction_intervals[:, 0]) & (y_val <= prediction_intervals[:, 1]))
average_length = np.mean(prediction_intervals[:, 1] - prediction_intervals[:, 0])

print(f'Coverage: {coverage}')
print(f'Average Length: {average_length}')

# Have the issue of having negative lower bounds in the intervals

