# GBDT — high-level pseudocode
# INPUT  Common training, calibration, and retrospective test partitions.
# PREPARE  Prepare the adsorption descriptors for the boosting model.
# TRAIN  Fit regression trees sequentially to reduce the current prediction error.
# PREDICT  Combine the fitted trees' contributions to estimate uptake.
# EVALUATE  Compare predictions with measured uptake using RMSE, MAE, and R².
# OUTPUT  Uptake predictions and evaluation scores for the designated partitions.
