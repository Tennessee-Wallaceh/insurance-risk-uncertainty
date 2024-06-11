#%%
import pandas as pd
import numpy as np

from xgboostlss.model import *
from xgboostlss.distributions.ZIPoisson import *

from scipy.stats import poisson
import plotnine
from plotnine import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#%%
np.random.seed(123)
#%%
# load data.csv
data = pd.read_csv('data_processed.csv', index_col=0)

# %%
# Feature engeneering done by Sam
# - truncate the number of claims to 4 
data["ClaimNb"] = np.where(data["ClaimNb"] > 4, 4, data["ClaimNb"])
# - one hot encoding of Area 
# - trucate VehPower to 9 
# - bind age to 7 groups instead of treating it as continous 
# - truncate BonusMalus to 150 
# - log transform Density
#%%
# truncate exposure to 1
data["Exposure"] = np.where(data["Exposure"] > 1, 1, data["Exposure"])
#%%
# divide the number of claims by the exposure
data["Frequency"] = data["ClaimNb"] / data["Exposure"]
#%%
# round the frequency to the nearest integer
data["Frequency"] = np.round(data["Frequency"])
#%%
# plot an histogram of the frequency with matplotlib removing the zeros
# and the values above 50
plt.hist(data["Frequency"], bins=50, range=(1, 50))
plt.show()
#%%
# categorical variables 
categorical_columns = ['VehBrand', 'VehGas', 'Area', 'Region']
for col in categorical_columns:
    data[col] = data[col].astype('category')
#%%
# split data into train, calibration and test so that 
# teh calibration is 5% of the data and the test is 20% of the data
train = data.sample(frac=0.75)
data = data.drop(train.index)
calibration = data.sample(frac=0.05)
test = data.drop(calibration.index)
#%% 
#reorder row names
train = train.reset_index(drop=True)
calibration = calibration.reset_index(drop=True)
test = test.reset_index(drop=True)
#%%
# set up data for XGBoost 
y = train["Frequency"]
X = train.drop(columns=["Frequency", "ClaimNb", "Exposure",
                       "AreaGLM", "VehPowerGLM","VehAgeGLM",
                        "DrivAgeGLM", "BonusMalusGLM", "DensityGLM" ])
dtrain = DMatrix(data=X, label=y, enable_categorical=True)

#%%
# set up the calibration data
y_cal = calibration["Frequency"]
X_cal = calibration.drop(columns=["Frequency", "ClaimNb", "Exposure",
                            "AreaGLM", "VehPowerGLM","VehAgeGLM",
                            "DrivAgeGLM", "BonusMalusGLM", "DensityGLM" ])
dcal = DMatrix(data=X_cal, label=y_cal, enable_categorical=True)

#%%

# set up the test data
y_test = test["Frequency"]
X_test = test.drop(columns=["Frequency", "ClaimNb", "Exposure",
                            "AreaGLM", "VehPowerGLM","VehAgeGLM",
                            "DrivAgeGLM", "BonusMalusGLM", "DensityGLM" ])
dtest = DMatrix(data=X_test, label=y_test, enable_categorical=True)

#%%
xgblss = XGBoostLSS(
    ZIPoisson(stabilization='None', 
            response_fn="exp", 
            loss_fn="nll")
)
#%%
param_dict = {
    "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
    "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
    "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
    "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
    "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
    "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}],
    "booster":          ["categorical", ["gbtree"]]
}

#%%
opt_param = xgblss.hyper_opt(param_dict,
                             dtrain,
                             num_boost_round=100,        # Number of boosting iterations.
                             nfold=5,                    # Number of cv-folds.
                             early_stopping_rounds=20,   # Number of early-stopping rounds
                             max_minutes=10,             # Time budget in minutes, i.e., stop study after the given number of minutes.
                             n_trials=30 ,               # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
                             silence=True,               # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
                             seed=123,                   # Seed used to generate cv-folds.
                             hp_seed=123                 # Seed for random number generator used in the Bayesian hyperparameter search.
                            )
#%%
opt_params = opt_param.copy()
n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]

# Train Model with optimized hyperparameters
xgblss.train(opt_params,
             dtrain,
             num_boost_round=n_rounds
             )
# %%
# FIXED SIZE PREDICITON INTERVALS   
# Make predictions on calibration data
preds_cal_par = xgblss.predict(dcal)
# pred = (1-gate)*rate
preds_cal =  preds_cal_par["gate"] * preds_cal_par["rate"]

#%%
# Make predictions on test data
preds_test_par = xgblss.predict(dtest)
# pred = (1-gate)*rate
preds_test =  preds_test_par["gate"] * preds_test_par["rate"]
#%%
# Calculate residuals (nonconformity scores) on calibration data
calibration["residuals"] = np.abs(calibration["Frequency"] - preds_cal)

#%%
# Determine the quantile for the desired confidence level
confidence_level = 0.95
alpha = 1 - confidence_level
q_hat = np.quantile(calibration["residuals"], 1 - alpha)

#%%
# Compute prediction intervals
test["lower_bound"] = preds_test - q_hat
test["upper_bound"] = preds_test + q_hat

#%%

# Optionally, display the prediction intervals for the first few rows
print(test[["Frequency", "lower_bound", "upper_bound"]].head())
# plot the prediction intervals
#%%
# plot the prediction intervals
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# Plot the true frequencies in red
plt.plot(test.index, test["Frequency"], "o", color='red', label="True Frequency")
# Plot the predicted frequencies in blue
plt.plot(test.index, preds_test, "o", color='blue', label="Predicted Frequency")
# Add the prediction intervals as vertical lines
plt.vlines(test.index, test["lower_bound"], test["upper_bound"], color="gray", alpha=0.5, label="95% Prediction Interval")
plt.xlabel("Index")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, preds_test)

