# Deep-MLP — high-level pseudocode
# INPUT  Common training, calibration, and retrospective test partitions.
# PREPARE  Represent adsorption descriptors as inputs to a multilayer network.
# TRAIN  Fit a multilayer neural regressor to the measured uptake.
# PREDICT  Use the fitted network to estimate uptake.
# EVALUATE  Compare predictions with measured uptake using RMSE, MAE, and R².
# OUTPUT  Uptake predictions and evaluation scores for the designated partitions.
