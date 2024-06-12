#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
#%%
# load dara from the csv file
info_train = pd.read_csv('info_train.csv', index_col=0)
info_cal = pd.read_csv('info_cal.csv', index_col=0)
info_test = pd.read_csv('info_test.csv', index_col=0)

#%%
# truncate exposure to 1
info_train["Exposure"] = np.where(info_train["Exposure"] > 1, 1, info_train["Exposure"])
info_cal["Exposure"] = np.where(info_cal["Exposure"] > 1, 1, info_cal["Exposure"])
info_test["Exposure"] = np.where(info_test["Exposure"] > 1, 1, info_test["Exposure"])

#%%

# Trying scaled Nb with Exposure
info_train['Scaled_ClaimNb'] = info_train['ClaimNb_cap'] / info_train['Exposure']
info_cal['Scaled_ClaimNb'] = info_cal['ClaimNb_cap'] / info_cal['Exposure']
info_test['Scaled_ClaimNb'] = info_test['ClaimNb_cap'] / info_test['Exposure']
#%%

# Log transformation and filtering
info_train['log_Scaled_ClaimNb'] = np.log(info_train['Scaled_ClaimNb'] + 1)
info_cal['log_Scaled_ClaimNb'] = np.log(info_cal['Scaled_ClaimNb'] + 1)
info_test['log_Scaled_ClaimNb'] = np.log(info_test['Scaled_ClaimNb'] + 1)
#%%
#%%
# Encoding categorical features
categorical_features = ['Area', 'VehBrand', 'VehGas', 'Region']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    info_train[feature] = le.fit_transform(info_train[feature])
    info_cal[feature] = le.transform(info_cal[feature])
    info_test[feature] = le.transform(info_test[feature])
    label_encoders[feature] = le


#%%


# Prepare data for XGBoost
X_train = info_train.iloc[:, 2:11]
# y_train = info_train['Scaled_ClaimNb']
y_train = info_train['log_Scaled_ClaimNb']

X_cal = info_cal.iloc[:, 2:11]
# y_cal = info_cal['Scaled_ClaimNb']
y_cal = info_cal['log_Scaled_ClaimNb']

X_test = info_test.iloc[:, 2:11]
# y_test = info_test['Scaled_ClaimNb']
y_test = info_test['log_Scaled_ClaimNb']
#%%
# histogram of the target variable
plt.hist(y_test, bins=30)
plt.show()

#%%

# Assuming X_train, X_test, y_train, y_test are already defined
# Converting to DMatrix for XGBoost
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dcal = xgb.DMatrix(data=X_cal, label=y_cal)
dtest = xgb.DMatrix(data=X_test, label=y_test)
# XGBoost model parameters
params = {
    'booster': 'gbtree',
    'objective': 'reg:squaredlogerror',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.5,
    'colsample_bytree': 0.7
}

# XGBoost model
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100, verbose_eval=False)

#%%
def calculate_gini(actual, predicted):
    df = pd.DataFrame({'actual': actual, 'predicted': predicted})
    df = df.sort_values('predicted')

    # Calculate the cumulative sums of actual values
    cum_actuals = np.cumsum(df['actual']) / sum(df['actual'])
    cum_predicted = np.cumsum(df['predicted']) / sum(df['predicted'])

    # Area under the Lorenz curve
    Lorenz = np.cumsum(np.sort(df['actual']) / sum(df['actual']))
    B = sum(Lorenz[:-1]) / (len(Lorenz) - 1)

    # Area above the Lorenz curve
    A = 0.5 - B
    gini = A / 0.5

    return gini

# XGBoost predictions
predictions_xgb = xgb_model.predict(dtest)

# Plot histogram of predictions
plt.hist(predictions_xgb, bins=30)
plt.show()

#%%
# MSE for XGBoost
mse_xgb = mean_squared_error(y_test, predictions_xgb)
print(f"XGBoost MSE: {mse_xgb}")

#%%
# Gini for XGBoost
gini_xgb = calculate_gini(y_test, predictions_xgb)
print(f"XGBoost Gini: {gini_xgb}")

#%%
# plot y_test vs predictions
plt.scatter(y_test, predictions_xgb)
plt.xlabel('True values')
plt.ylabel('Predictions')
#%%
# quantile regression 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error

all_models = {}
common_params = dict(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=9,
    min_samples_split=9,
)
for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)

gbr_ls = GradientBoostingRegressor(loss="squared_error", **common_params)
all_models["mse"] = gbr_ls.fit(X_train, y_train)

#%%
def highlight_min(x):
    x_min = x.min()
    return ["font-weight: bold" if v == x_min else "" for v in x]


results = []
for name, gbr in sorted(all_models.items()):
    metrics = {"model": name}
    y_pred = gbr.predict(X_test)
    for alpha in [0.05, 0.5, 0.95]:
        metrics["pbl=%1.2f" % alpha] = mean_pinball_loss(y_test, y_pred, alpha=alpha)
    metrics["MSE"] = mean_squared_error(y_test, y_pred)
    results.append(metrics)

pd.DataFrame(results).set_index("model").style.apply(highlight_min)
# %%
y_low = all_models["q 0.05"].predict(X_test)
y_high= all_models["q 0.95"].predict(X_test)

# %%
print(np.mean(np.logical_and(y_test >= y_low, y_test <= y_high)))
print(np.mean(y_test ==0))
# %%
# not wroking, the predicitons intervals are all zero. 