# RandomForest — high-level pseudocode
# INPUT  Common training, calibration, and retrospective test partitions.
# PREPARE  Prepare the adsorption descriptors for a regression forest.
# TRAIN  Fit trees using resampled training observations and randomized feature choices.
# PREDICT  Average the trees' uptake estimates.
# EVALUATE  Compare predictions with measured uptake using RMSE, MAE, and R².
# OUTPUT  Uptake predictions and evaluation scores for the designated partitions.
