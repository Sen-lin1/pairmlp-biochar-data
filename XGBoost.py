# XGBoost — high-level pseudocode
# INPUT  Common training, calibration, and retrospective test partitions.
# PREPARE  Prepare the adsorption descriptors for the boosting model.
# TRAIN  Fit boosted trees with the XGBoost method to reduce prediction error.
# PREDICT  Combine the fitted trees' contributions to estimate uptake.
# EVALUATE  Compare predictions with measured uptake using RMSE, MAE, and R².
# OUTPUT  Uptake predictions and evaluation scores for the designated partitions.
