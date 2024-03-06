import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np

# Load the dataset
wine_data = pd.read_csv('wine_dataset.csv')

# Assuming 'style' is the target variable, separate the features and the target
wine_target = wine_data['style']
wine_features = wine_data.drop('style', axis=1)

# Convert target to binary encoding
wine_target = wine_target.map({'red': 0, 'white': 1})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    wine_features, wine_target, test_size=0.3, random_state=1, stratify=wine_target)

# Standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Fit a logistic regression model with no regularization
lr_none = LogisticRegression(penalty=None, random_state=1)
lr_none.fit(X_train_std, y_train)

# Fit logistic regression models with L1 penalty
C_values = [0.001, 0.01, 0.1, 1, 10]  #regularization strength
lr_l1_models = [LogisticRegression(penalty='l1', C=C_val, solver='liblinear', random_state=1).fit(X_train_std, y_train) for C_val in C_values]

# Fit logistic regression models with L2 penalty
lr_l2_models = [LogisticRegression(penalty='l2', C=C_val, solver='liblinear', random_state=1).fit(X_train_std, y_train) for C_val in C_values]

# Evaluate the models on the test set
print("Model with no regularization score:", lr_none.score(X_test_std, y_test))
for i, C_val in enumerate(C_values):
    print(f"L1 regularized model with C={C_val} score:", lr_l1_models[i].score(X_test_std, y_test))
    print(f"L2 regularized model with C={C_val} score:", lr_l2_models[i].score(X_test_std, y_test))

# (b)
weights = lr_none.coef_
bias = lr_none.intercept_
l2_norm = np.sqrt(np.sum(weights**2))
print(f'weights: {weights}')
print(f'L2 norm of the weights: {l2_norm}')

# (c)
# Evaluate the models on the test set and store the scores
scores = {'none': lr_none.score(X_test_std, y_test)}
scores_l1 = [model.score(X_test_std, y_test) for model in lr_l1_models]
scores_l2 = [model.score(X_test_std, y_test) for model in lr_l2_models]

print("Model with no regularization score:", scores['none'])

# Print the scores and calculate L2 norms for the L1 and L2 models
for i, C_val in enumerate(C_values):
    print(f"L1 regularized model with C={C_val} score:", scores_l1[i])
    print(f"L2 regularized model with C={C_val} score:", scores_l2[i])
    scores[f'l1_{C_val}'] = scores_l1[i]
    scores[f'l2_{C_val}'] = scores_l2[i]

# Calculate the L2 norm of the weights for the model with no regularization
weights_none = lr_none.coef_
l2_norm_none = np.linalg.norm(weights_none)
print(f'L2 norm of the weights (no regularization): {l2_norm_none}')

# Find the highest accuracy L1 and L2 models and their L2 norms
best_l1_index = np.argmax(scores_l1)
best_l2_index = np.argmax(scores_l2)
best_l1_model = lr_l1_models[best_l1_index]
best_l2_model = lr_l2_models[best_l2_index]

weights_l1_best = best_l1_model.coef_
weights_l2_best = best_l2_model.coef_
l2_norm_l1_best = np.linalg.norm(weights_l1_best)
l2_norm_l2_best = np.linalg.norm(weights_l2_best)

print(f'L2 norm of the weights (best L1 model): {l2_norm_l1_best}')
print(f'L2 norm of the weights (best L2 model): {l2_norm_l2_best}')

# Count the number of zero weights in the models
zero_weights_none = np.sum(weights_none == 0)
zero_weights_l1_best = np.sum(weights_l1_best == 0)
zero_weights_l2_best = np.sum(weights_l2_best == 0)

print(f'Number of zero weights (no regularization): {zero_weights_none}')
print(f'Number of zero weights (best L1 model): {zero_weights_l1_best}')
print(f'Number of zero weights (best L2 model): {zero_weights_l2_best}')