import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from init import import_dataset
from sklearn.metrics import mean_squared_error

import collections

#############################################
#                  Del 1                    #
#############################################

# Load data file and extract variables of interest

df = import_dataset(2016)

# Binary target value
y = df['Happiness Score']
X = df[['Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia Residual']].copy()
# Standardize the training and test set based on training set mean and std

for idx in range(X.shape[1]):
    mu = np.mean(X.iloc[:,idx], 0)
    sigma = np.std(X.iloc[:,idx], 0)

    X.iloc[:,idx] = (X.iloc[:,idx] - mu) / sigma

#############################################
#                  Del 2                    #
#############################################

# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K,shuffle=True)

overall_optimal_lambdas = {}
overall_test_errors = {}

# Here we determine the range for our regulisation term Lambda.
lambda_interval = np.logspace(-3, 1, 50)
idx = 0 # Index value for optimal_lmabdas dictionary.

# Final model Coefs
predictions = []

for train_index, test_index in CV.split(X,y):
    idx += 1

    print(f"Cross Validation split: {idx}")

    X_train_outer, y_train_outer = X.iloc[train_index,:], y[train_index]
    X_test_outer, y_test_outer = X.iloc[test_index,:], y[test_index]

    # Doing the 2nd level partition of the train data.
    X_train_inner, X_test_inner, y_train_inner, y_test_inner = train_test_split(X_train_outer, y_train_outer, train_size=0.75)

    test_error_rate = np.zeros(len(lambda_interval))


    for k in range(0, len(lambda_interval)):
        mdl = Ridge(alpha=lambda_interval[k] )

        mdl.fit(X_train_inner, y_train_inner)

        y_test_est = mdl.predict(X_test_inner).T
        
        test_error_rate[k] = mean_squared_error(y_test_est, y_test_inner)


    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]
    overall_optimal_lambdas[idx] = opt_lambda

    print(f"For split number: {idx}, Optimal lambda found to be: {opt_lambda}")

    # Fit new model based on the original test split / partition by the K-fold.
    # We will use the optimal lambda we just found.
    mdl_original = Ridge(alpha=opt_lambda )
    mdl_original.fit(X_train_outer, y_train_outer)
    y_train_est_outer = mdl_original.predict(X_train_outer).T
    y_test_est_outer = mdl_original.predict(X_test_outer).T
    # Add the test_error_rate to the overall test_error_rate so we can calculate the generalisation error in the end.
    overall_test_errors[idx] = mean_squared_error(y_test_est_outer, y_test_outer)

    predictions.append(y_test_est_outer.tolist())

mean_overall_error = sum(overall_test_errors.values()) / len(overall_test_errors.values())
print(f"Mean Overall MSError (Generalisation Error): {mean_overall_error}")

