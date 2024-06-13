# Gamma model ----
# Fit the Gamma regression model
glm_sev <- glm(ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus 
               + VehBrand + VehGas + Density + Region + Area, 
               family = Gamma(link = "log"), 
               data = comb_tr_train)

# Generate predictions for the training set
comb_tr_train$predicted <- predict(glm_sev, newdata = comb_tr_train, type = "response")

# Generate predictions for the testing set
comb_tr_test$predicted <- predict(glm_sev, newdata = comb_tr_test, type = "response")

# Deviance and null deviance for train set
deviance_train <- sum((comb_tr_train$ClaimAmount - comb_tr_train$predicted)^2)
null_deviance_train <- sum((comb_tr_train$ClaimAmount - mean(comb_tr_train$ClaimAmount))^2)

# Deviance and null deviance for test set
deviance_test <- sum((comb_tr_test$ClaimAmount - comb_tr_test$predicted)^2)
null_deviance_test <- sum((comb_tr_test$ClaimAmount - mean(comb_tr_test$ClaimAmount))^2)

# D² explained
D2_train <- 1 - deviance_train / null_deviance_train
D2_test <- 1 - deviance_test / null_deviance_test

# Mean Absolute Error (MAE)
MAE_train <- mean(abs(comb_tr_train$ClaimAmount - comb_tr_train$predicted))
MAE_test <- mean(abs(comb_tr_test$ClaimAmount - comb_tr_test$predicted))

# Mean Squared Error (MSE)
MSE_train <- mean((comb_tr_train$ClaimAmount - comb_tr_train$predicted)^2)
MSE_test <- mean((comb_tr_test$ClaimAmount - comb_tr_test$predicted)^2)

# Create a data frame to store the metrics
metrics <- data.frame(
  Subset = c("Train", "Test"),
  `D² Explained` = c(D2_train, D2_test),
  `Mean Absolute Error` = c(MAE_train, MAE_test),
  `Mean Squared Error` = c(MSE_train, MSE_test)
)

# Print the table using knitr::kable
kable(metrics, format = "markdown", col.names = c("Subset", "D² Explained", "Mean Absolute Error", "Mean Squared Error"))








# Gamma model ----
# Fit the Gamma regression model
glm_sev <- glm(ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus 
               + VehBrand + VehGas + Density + Region + Area, 
               family = Gamma(link = "log"), 
               data = comb_train)

# Generate predictions for the training set
comb_train$predicted <- predict(glm_sev, newdata = comb_train, type = "response")

# Generate predictions for the testing set
comb_test$predicted <- predict(glm_sev, newdata = comb_test, type = "response")

# Deviance and null deviance for train set
deviance_train <- sum((comb_train$ClaimAmount - comb_train$predicted)^2)
null_deviance_train <- sum((comb_train$ClaimAmount - mean(comb_train$ClaimAmount))^2)

# Deviance and null deviance for test set
deviance_test <- sum((comb_test$ClaimAmount - comb_test$predicted)^2)
null_deviance_test <- sum((comb_test$ClaimAmount - mean(comb_test$ClaimAmount))^2)

# D² explained
D2_train <- 1 - deviance_train / null_deviance_train
D2_test <- 1 - deviance_test / null_deviance_test

# Mean Absolute Error (MAE)
MAE_train <- mean(abs(comb_train$ClaimAmount - comb_train$predicted))
MAE_test <- mean(abs(comb_test$ClaimAmount - comb_test$predicted))

# Mean Squared Error (MSE)
MSE_train <- mean((comb_train$ClaimAmount - comb_train$predicted)^2)
MSE_test <- mean((comb_test$ClaimAmount - comb_test$predicted)^2)

# Create a data frame to store the metrics
metrics <- data.frame(
  Subset = c("Train", "Test"),
  `D² Explained` = c(D2_train, D2_test),
  `Mean Absolute Error` = c(MAE_train, MAE_test),
  `Mean Squared Error` = c(MSE_train, MSE_test)
)

# Print the table using knitr::kable
kable(metrics, format = "markdown", col.names = c("Subset", "D² Explained", "Mean Absolute Error", "Mean Squared Error"))
