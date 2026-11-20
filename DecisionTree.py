# DecisionTree — high-level pseudocode
# INPUT  Common training, calibration, and retrospective test partitions.
# PREPARE  Prepare the adsorption descriptors for a single regression tree.
# TRAIN  Fit a tree that partitions training observations by their descriptors.
# PREDICT  Follow the fitted tree to a terminal node and return its uptake estimate.
# EVALUATE  Compare predictions with measured uptake using RMSE, MAE, and R².
# OUTPUT  Uptake predictions and evaluation scores for the designated partitions.
