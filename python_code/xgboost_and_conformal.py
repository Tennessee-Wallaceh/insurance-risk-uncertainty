# Trying point estimation on the claim amoount using standard xgboost

import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import norm
import matplotlib.pyplot as plt


# In previous script I have experimented with XGboost
# In this script I will try to use XGboost with Conformal Prediction

# Load data
# data = pd.read_csv('merged_data.csv')
comb_tr_train =pd.read_csv('Data/comb_tr_train.csv')
# remove first column
comb_tr_train = comb_tr_train.drop(columns=['Unnamed: 0'])

categorical_columns = ['VehPower', 'VehAge', 'VehBrand', 'VehGas', 'Area', 'Region','DrivAge']
for col in categorical_columns:
    comb_tr_train[col] = comb_tr_train[col].astype('category')

# print column names
print(comb_tr_train.columns)
X_train = comb_tr_train.drop(columns=['IDpol','ClaimAmount', 'Exposure', 'ClaimNb', 'ClaimNb_cap'])
y_train = comb_tr_train['ClaimAmount']

# Split data into train+calibration and validation sets
# X_train_cal, X_val, y_train_cal, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train+calibration set into training and calibration sets
#X_train, X_cal, y_train, y_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

####################################
####################### hyperparameter tuning #######################
param_grid = {
    'colsample_bytree': [0.3, 0.5, 0.7],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 15],
    'alpha': [1, 10, 100],
    'n_estimators': [50, 100, 200, 500]
}

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical=True) #MAE: 496 #MSE: 631109.88 #Max:3749.6
xgb_reg = xgb.XGBRegressor(objective='reg:gamma', enable_categorical=True) #MAE: 486.59  #MSE:616564.349 #Max: 3694


# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_reg, param_distributions=param_grid, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Perform the search
random_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)

# Train the model with the best parameters
best_params = random_search.best_params_
best_xgb_reg = xgb.XGBRegressor(objective='reg:gamma', **best_params, enable_categorical=True)
#best_xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', **best_params, enable_categorical=True)
best_xgb_reg.fit(X_train, y_train)

####################################################################


#####################################################################
## loading in claibration data ##
comb_tr_cal =pd.read_csv('Data/comb_tr_cal.csv')
# remove first column
comb_tr_cal = comb_tr_cal.drop(columns=['Unnamed: 0'])

categorical_columns = ['VehPower', 'VehAge', 'VehBrand', 'VehGas', 'Area', 'Region', 'DrivAge']
for col in categorical_columns:
    comb_tr_cal[col] = comb_tr_cal[col].astype('category')


# print column names
print(comb_tr_cal.columns)
X_cal = comb_tr_cal.drop(columns=['IDpol','ClaimAmount', 'Exposure', 'ClaimNb', 'ClaimNb_cap'])
y_cal = comb_tr_cal['ClaimAmount']



# Split data into train+calibration and validation sets
# X_train_cal, X_val, y_train_cal, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train+calibration set into training and calibration sets
#X_train, X_cal, y_train, y_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

dcal= xgb.DMatrix(X_cal, label=y_cal, enable_categorical=True)


# compute the score function
# Make predictions on the calibration set
cal_preds = best_xgb_reg.predict(X_cal)

# Calculate the absolute errors (for regression)
cal_errors = np.abs(cal_preds - y_cal)


# get the quantile
alpha = 0.1
quantile = np.quantile(cal_errors, 1 - alpha)



#####################################################################
# Make predictions on the validation set
## loading in testing data ##
comb_tr_test =pd.read_csv('Data/comb_tr_test.csv')
# remove first column
comb_tr_test = comb_tr_test.drop(columns=['Unnamed: 0'])

categorical_columns = ['VehPower', 'VehAge', 'VehBrand', 'VehGas', 'Area', 'Region', 'DrivAge']
for col in categorical_columns:
    comb_tr_test[col] = comb_tr_test[col].astype('category')


# print column names
print(comb_tr_test.columns)
X_val = comb_tr_test.drop(columns=['IDpol','ClaimAmount', 'Exposure', 'ClaimNb', 'ClaimNb_cap'])
y_val = comb_tr_test['ClaimAmount']
val_preds = best_xgb_reg.predict(X_val)

# Model performance, MAE and MSE:
print(f'MAE: {np.mean(np.abs(val_preds - y_val))}')
print(f'MSE: {mean_squared_error(y_val, val_preds)}')

#########################################################################
# Create prediction intervals
prediction_intervals = np.array([
    (pred - quantile, pred + quantile) for pred in val_preds
])

# Calculate coverage and average length of the intervals
coverage = np.mean((y_val >= prediction_intervals[:, 0]) & (y_val <= prediction_intervals[:, 1]))
average_length = np.mean(prediction_intervals[:, 1] - prediction_intervals[:, 0])

print(f'Coverage: {coverage}')
print(f'Average Length: {average_length}')
# plot histogram of the interval sizes:
plt.hist(prediction_intervals[:, 1] - prediction_intervals[:, 0], bins=20)
plt.xlabel('Interval Size')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Interval Sizes')
plt.show()
# Have the issue of having negative lower bounds in the intervals
##########################################################################


