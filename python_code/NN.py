#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import torch

# Load data from the CSV file
info_train = pd.read_csv('info_train.csv', index_col=0)
info_cal = pd.read_csv('info_cal.csv', index_col=0)
info_test = pd.read_csv('info_test.csv', index_col=0)

# Make all the column names lowercase
info_train.columns = info_train.columns.str.lower()
info_cal.columns = info_cal.columns.str.lower()
info_test.columns = info_test.columns.str.lower()

# Mappers
for df in [info_train, info_cal, info_test]:
    mapper_vehpower = {j: str(int(j)) for j in df['vehpower'].unique()}
    mapper_vehbrand = {j: j for j in df['vehbrand'].unique()}
    mapper_vehgas = {j: j for j in df['vehgas'].unique()}
    mapper_area = {j: j for j in df['area'].unique()}
    mapper_region = {j: j for j in df['region'].unique()}

    # Bin edges for bandings
    bins_vehage = [-1, 0, 1, 4, 10, np.inf]
    bins_drivage = [0, 18, 21, 25, 35, 45, 55, 70, np.inf]

    # Helper function
    def transform_data(data):
        d = data.copy()
        # Location features
        d['area'] = d['area'].map(mapper_area).astype(str)
        d['region'] = d['region'].map(mapper_region).astype(str)
        d['density'] = np.log(d['density'])
        
        # Vehicle features
        d['vehpower'] = d['vehpower'].map(mapper_vehpower).astype(str)
        d['vehage'] = pd.cut(d['vehage'], bins=bins_vehage).astype(str)
        d['vehbrand'] = d['vehbrand'].map(mapper_vehbrand).astype(str)
        d['vehgas'] = d['vehgas'].map(mapper_vehgas).astype(str)
    
        # Driver features
        d['drivage'] = pd.cut(d['drivage'], bins=bins_drivage).astype(str)
        
        return d

    # Apply groupings and transformations
    df = transform_data(df)

# One-hot encoding for 'vehbrand', 'vehgas', and 'region'
info_train = pd.get_dummies(info_train, columns=['vehbrand', 'vehgas', 'region'], dtype=float)
info_cal = pd.get_dummies(info_cal, columns=['vehbrand', 'vehgas', 'region'], dtype=float)
info_test = pd.get_dummies(info_test, columns=['vehbrand', 'vehgas', 'region'], dtype=float)

# Ensure all columns are numeric
X_train = info_train.iloc[:, 2:11].astype(float)
y_train = info_train['claimnb_cap'].astype(float)
X_cal = info_cal.iloc[:, 2:11].astype(float)
y_cal = info_cal['claimnb_cap'].astype(float)
X_test = info_test.iloc[:, 2:11].astype(float)
y_test = info_test['claimnb_cap'].astype(float)

# Transform all the data to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_cal = torch.tensor(X_cal.values, dtype=torch.float32)
y_cal = torch.tensor(y_cal.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
train_evy = torch.tensor(info_train['exposure'].values, dtype=torch.float32)
cal_evy = torch.tensor(info_cal['exposure'].values, dtype=torch.float32)
test_evy = torch.tensor(info_test['exposure'].values, dtype=torch.float32)
# Compute the frequency of the claims
frequency = info_train['claimnb_cap'] / info_train['exposure']
y_hat = frequency.mean()
#%%
# Define the neural network model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        input_size = X_train.shape[1]  # Match the input size to the feature size
        self.F_hidden_one = torch.nn.Linear(input_size, 250)
        self.F_hidden_two = torch.nn.Linear(250, 250)
        self.F_output = torch.nn.Linear(250, 1)
        self.dropout_one = torch.nn.Dropout(p=0.25)
        self.dropout_two = torch.nn.Dropout(p=0.25)
        torch.nn.init.kaiming_uniform_(self.F_hidden_one.weight)
        torch.nn.init.kaiming_uniform_(self.F_hidden_two.weight)
        torch.nn.init.kaiming_uniform_(self.F_output.weight)
        torch.nn.init.constant_(self.F_output.bias, y_hat)
        
    def forward(self, x):
        elu = torch.nn.ELU(alpha=1)
        F = self.dropout_one(x)
        F = self.F_hidden_one(F)
        F = elu(F)
        F = self.dropout_two(F)
        F = self.F_hidden_two(F)
        F = elu(F)
        F = self.F_output(F)
        F = torch.exp(F)
        return F

#%%
# Initialize the model
model = NeuralNetwork().float()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
n_epoch = 10

def poisson_loss(y_true, y_pred):
    """
    Poisson loss function for insurance claims prediction.

    Args:
    y_true (Tensor): The actual number of claims.
    y_pred (Tensor): The predicted number of claims.
    evy (Tensor): The exposure values.

    Returns:
    Tensor: The computed Poisson loss.
    """
    return torch.mean(y_pred - y_true * torch.log(y_pred))

# Training loop
for epoch in range(1, n_epoch + 1):
    # Set model to training mode
    model.train()
    # Make predictions
    F = model(X_train)
    # back-prop & weight update
    train_loss = poisson_loss(y_train,train_evy.float() * F)
    train_loss.backward()
    optimizer.zero_grad()
    optimizer.step()

    # print(f"Epoch {epoch}, Loss: {loss.item()}")

# %%
