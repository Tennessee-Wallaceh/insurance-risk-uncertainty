import pandas as pd
import numpy as np
from statsmodels.discrete.count_model import ZeroInflatedPoisson
import statsmodels.api as sm
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your dataset
info_tr_train = pd.read_csv('Data/info_tr_train.csv')
# print column names
print(info_tr_train.columns)

# Assuming 'claims' is the target variable and the rest are features
X_train= info_tr_train.drop('ClaimNb_cap', axis=1)
y_train = info_tr_train['ClaimNb_cap']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the features matrix for statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit a Zero-Inflated Poisson model without resampling
zip_model = ZeroInflatedPoisson(y_train, X_train_sm, exog_infl=X_train_sm, inflation='logit')
zip_results = zip_model.fit()

# Predictions and evaluation
y_pred = zip_results.predict(X_test_sm)
predicted_probabilities = zip_results.predict(X_test_sm, which='mean')

# Filter out the nonzero predictions and actuals
y_test_nonzero = y_test[y_test > 0]
y_pred_nonzero = y_pred[y_test > 0]

# Calculate precision, recall, and F1 score for nonzero predictions
precision_nonzero = precision_score(y_test_nonzero > 0, y_pred_nonzero > 0)
recall_nonzero = recall_score(y_test_nonzero > 0, y_pred_nonzero > 0)
f1_nonzero = f1_score(y_test_nonzero > 0, y_pred_nonzero > 0)

print("Performance without resampling:")
print(f"Precision (nonzero): {precision_nonzero}")
print(f"Recall (nonzero): {recall_nonzero}")
print(f"F1 Score (nonzero): {f1_nonzero}")

# Combine over-sampling and under-sampling using SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# Add a constant to the resampled features matrix for statsmodels
X_resampled_sm = sm.add_constant(X_resampled)

# Fit a Zero-Inflated Poisson model with resampling
zip_model_resampled = ZeroInflatedPoisson(y_resampled, X_resampled_sm, exog_infl=X_resampled_sm, inflation='logit')
zip_results_resampled = zip_model_resampled.fit()

# Predictions and evaluation on the original test set
y_pred_resampled = zip_results_resampled.predict(X_test_sm)
predicted_probabilities_resampled = zip_results_resampled.predict(X_test_sm, which='mean')

# Filter out the nonzero predictions and actuals
y_pred_resampled_nonzero = y_pred_resampled[y_test > 0]

# Calculate precision, recall, and F1 score for nonzero predictions
precision_nonzero_resampled = precision_score(y_test_nonzero > 0, y_pred_resampled_nonzero > 0)
recall_nonzero_resampled = recall_score(y_test_nonzero > 0, y_pred_resampled_nonzero > 0)
f1_nonzero_resampled = f1_score(y_test_nonzero > 0, y_pred_resampled_nonzero > 0)

print("Performance with resampling:")
print(f"Precision (nonzero): {precision_nonzero_resampled}")
print(f"Recall (nonzero): {recall_nonzero_resampled}")
print(f"F1 Score (nonzero): {f1_nonzero_resampled}")

# Summarize the results
print("\nSummary:")
print("Without Resampling")
print(zip_results.summary())

print("\nWith Resampling")
print(zip_results_resampled.summary())