import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'wine_dataset.csv'  # Replace with the path to your dataset
wine_data = pd.read_csv(file_path)

# Encode the 'style' column into a numerical format
label_encoder = LabelEncoder()
wine_data['style'] = label_encoder.fit_transform(wine_data['style'])

# Standardize the features
features = wine_data.drop('style', axis=1)
labels = wine_data['style']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into a 50/50 training and test set
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.5, random_state=1)

# Fit a logistic regression model to the training data
logistic_model = LogisticRegression(random_state=1)
logistic_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred_logistic = logistic_model.predict(X_test)

# Calculate the accuracy for the logistic regression model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

# Initialize SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', random_state=1)
svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test)
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)

# Try different gamma values for RBF kernel
gammas = [0.001, 0.01, 0.1, 1, 10]
svm_accuracies = {}
for gamma in gammas:
    svm_rbf_gamma = SVC(kernel='rbf', gamma=gamma, random_state=1)
    svm_rbf_gamma.fit(X_train, y_train)
    y_pred_svm_rbf_gamma = svm_rbf_gamma.predict(X_test)
    svm_accuracies[gamma] = accuracy_score(y_test, y_pred_svm_rbf_gamma)

# Try linear kernel
svm_linear = SVC(kernel='linear', random_state=1)
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)
accuracy_svm_linear = accuracy_score(y_test, y_pred_svm_linear)

# Try polynomial kernel
svm_poly = SVC(kernel='poly', random_state=1)
svm_poly.fit(X_train, y_train)
y_pred_svm_poly = svm_poly.predict(X_test)
accuracy_svm_poly = accuracy_score(y_test, y_pred_svm_poly)

# Print out the accuracies
print("Logistic Regression Accuracy:", accuracy_logistic)
print("SVM RBF Kernel Accuracy:", accuracy_svm_rbf)
print("SVM RBF Kernel Accuracies for different gammas:", svm_accuracies)
print("SVM Linear Kernel Accuracy:", accuracy_svm_linear)
print("SVM Polynomial Kernel Accuracy:", accuracy_svm_poly)
