# Pair-MLP — high-level pseudocode
# INPUT  Common training, calibration, and retrospective test partitions.
# PREPARE  Represent each biochar–pollutant pair alongside its adsorption descriptors.
# TRAIN  Fit a pair-aware neural regressor to the measured uptake.
# PREDICT  Use the fitted pair-aware regressor to estimate uptake.
# EVALUATE  Compare predictions with measured uptake using RMSE, MAE, and R².
# OUTPUT  Uptake predictions and evaluation scores for the designated partitions.
