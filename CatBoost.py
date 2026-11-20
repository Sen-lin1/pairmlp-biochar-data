# CatBoost — high-level pseudocode
# INPUT  Common training, calibration, and retrospective test partitions.
# PREPARE  Prepare numerical and categorical adsorption descriptors.
# TRAIN  Fit a categorical-aware boosted-tree regressor to measured uptake.
# PREDICT  Apply the fitted CatBoost regressor to estimate uptake.
# EVALUATE  Compare predictions with measured uptake using RMSE, MAE, and R².
# OUTPUT  Uptake predictions and evaluation scores for the designated partitions.
