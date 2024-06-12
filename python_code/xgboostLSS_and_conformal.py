# Trying xgboostlss on the claim amoount using standard to estimate the predicted distribution
# And using quantiles as score function for conformal prediction

from xgboostlss.distributions import *
from xgboostlss.distributions.distribution_utils import DistributionClass
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


from xgboostlss.model import *
from xgboostlss.distributions.Gamma import *
import multiprocessing

# source the data_python_script.py file
# exec(open("data_python_script.py").read())

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


n_cpu = multiprocessing.cpu_count()

dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu, enable_categorical=True)




dcal = xgb.DMatrix(X_cal, label=y_cal, nthread=n_cpu,  enable_categorical=True)
dcal = xgb.DMatrix(X_cal, nthread=n_cpu,  enable_categorical=True)





# Specifies Gamma distribution with exp response function and option to stabilize Gradient/Hessian. Type ?Gamma for an overview.
xgblss = XGBoostLSS(
    Gamma(stabilization="L2",     # Options are "None", "MAD", "L2".
          response_fn="exp",      # Function to transform the concentration and rate parameters, e.g., "exp" or "softplus".
          loss_fn="nll"           # Loss function. Options are "nll" (negative log-likelihood) or "crps"(continuous ranked probability score).
         )
)

param_dict = {
    "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
    "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
    "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
    "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
    "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
    "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}],
    "booster":          ["categorical", ["gbtree"]],
    # "tree_method":    ["categorical", ["auto", "approx", "hist", "gpu_hist"]],
    # "gpu_id":         ["none", [0]]
}

np.random.seed(123)
opt_param = xgblss.hyper_opt(param_dict,
                             dtrain,
                             num_boost_round=100,        # Number of boosting iterations.
                             nfold=5,                    # Number of cv-folds.
                             early_stopping_rounds=20,   # Number of early-stopping rounds
                             max_minutes=5,              # Time budget in minutes, i.e., stop study after the given number of minutes.
                             n_trials=None,              # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
                             silence=False,              # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
                             seed=123,                   # Seed used to generate cv-folds.
                             hp_seed=None                # Seed for random number generator used in the Bayesian hyperparameter search.
                            )



# saved opt_param
opt_param = {'eta': 0.5074453080193694, 'max_depth': 1, 'gamma': 6.28540486708738e-08, 'subsample': 0.9993409721839004, 
             'colsample_bytree': 0.8947273087407867, 'min_child_weight': 0.11250323097150428, 'booster': 'gbtree', 'opt_rounds': 97}

xgblss = XGBoostLSS(
    Gamma(stabilization="L2",     # Options are "None", "MAD", "L2".
          response_fn="exp",      # Function to transform the concentration and rate parameters, e.g., "exp" or "softplus".
          loss_fn="nll"           # Loss function. Options are "nll" (negative log-likelihood) or "crps"(continuous ranked probability score).
         )
)


# Fit the model
np.random.seed(123)

opt_params = opt_param.copy()
n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]

# Train Model with optimized hyperparameters
xgblss.train(opt_params,
             dtrain,
             num_boost_round=n_rounds
             )

############################################################################################################################
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
dcal= xgb.DMatrix(X_cal, label=y_cal, enable_categorical=True)

# Compute the score function
# Want to use score functions based on the predicted distribution
  
  # Will first try base on quantiles as in section 2.2 of Bates paper

torch.manual_seed(123)

n_samples = 5000
quant_sel = [0.05, 0.95] # Quantiles to calculate from predicted distribution


# Calculate quantiles from predicted distribution
pred_quantiles = xgblss.predict(dcal,
                                pred_type="quantiles",
                                n_samples=n_samples,
                                quantiles=quant_sel)

# for each point in y_cal compute the max difference between the quantiles:
# max(quantile_0.95 - y_cal, y_cal - quantile_0.05)
cal_errors = np.minimum(pred_quantiles.iloc[:,1].values - y_cal.values, y_cal.values - pred_quantiles.iloc[:, 0].values)

n = X_cal.shape[0]
cal_quant = np.ceil((n+1)*(1-0.1))/n
q_hat = np.quantile(cal_errors, cal_quant)


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
dval = xgb.DMatrix(X_val, nthread=n_cpu,  enable_categorical=True)
val_preds = xgblss.predict(dval)
# to get the mean estimate need to divide the first column by the second column:
val_preds_mean = val_preds.iloc[:,0] / val_preds.iloc[:,1]

# Model performance, MAE and MSE:
print(f'MAE: {np.mean(np.abs(val_preds_mean - y_val))}')
print(f'MSE: {mean_squared_error(y_val, val_preds_mean)}')
print(f'Max: {np.max(np.abs(val_preds_mean - y_val))}')


#########################################################################

# for each x in the validation set, compute the intervals
pred_quantiles_val = xgblss.predict(dval,
                                    pred_type="quantiles",
                                    n_samples=n_samples,
                                    quantiles=quant_sel)
# subtract q_hat from first column and add to second column
pred_quantiles_val.iloc[:,0] = pred_quantiles_val.iloc[:,0] - q_hat
pred_quantiles_val.iloc[:,1] = pred_quantiles_val.iloc[:,1] + q_hat

# histogram of interval sizes
plt.hist(pred_quantiles_val.iloc[:, 1] - pred_quantiles_val.iloc[:, 0], bins=20)
plt.xlabel('Interval Size')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Interval Sizes')
plt.show()


# plot dval ClaimAmount against DrivAge with intervals
plt.figure()
plt.scatter(X_val['VehAge'], y_val)
plt.scatter(X_val['BonusMalus'], y_val, label='True')
plt.fill_between(X_val['BonusMalus'], pred_quantiles_val.iloc[:,0], pred_quantiles_val.iloc[:,1], alpha=0.5, label='Predicted')
plt.legend()
plt.show()
# plot the intervals of pred_quantiles_val against X_val['VehAge']
df = pd.DataFrame({
    'VehAge': X_val['VehAge'].reset_index(drop=True),
    'PredQuantiles': pred_quantiles_val.iloc[:,0].reset_index(drop=True)
})
# Sort the DataFrame by 'VehAge'
df_sorted = df.sort_values(by='VehAge')
plt.figure()
plt.plot(X_val['VehAge'][0:5,], pred_quantiles_val.iloc[:,0][0:5,])
plt.plot(df_sorted['VehAge'][0:5], df_sorted['PredQuantiles'][0:5])
plt.plot(X_val['VehAge'], pred_quantiles_val.iloc[:,1])
plt.show()







# Can also try score function based on predicted density as in section 2.4 of Bates paper
# Returns predicted distributional parameters
# I think concentration is the shape parameter 
pred_params = xgblss.predict(dcal,
                             pred_type="parameters")



