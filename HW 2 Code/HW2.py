import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn

df = pd.read_csv('/Users/ernest/Downloads/Math 156/Math156HW/HW 2 Code/wine+quality/winequality-red.csv', delimiter = ";")
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

# Loading the Data, response: wine quality, predictor: everything else 
X = df.loc[:, "volatile acidity":"alcohol"].values 
y = df['quality'].values 

# Split the data set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)
y_train = preprocessing.normalize(y_train)
y_test = preprocessing.normalize(y_test)


# Train the data using the normal equation to find the weights of the linear regression model 
X_train_b  = np.c_[np.ones((X_train.shape[0], 1)), X_train]
w = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

# Predicton on the training set 
y_train_pred = X_train_b.dot(w)

plt.scatter(y_train, y_train_pred, alpha = 0.6) 
plt.title("Target vs. Predictions over Training Set")
plt.legend() 
plt.grid(True)
plt.show()

# Find the RMSE for the training set 
RMSE_train = mean_squared_error(y_train, y_train_pred, squared=False)
print(RMSE_train)

# Find the RMSE for the testing set 
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_test_pred = X_test_b.dot(w)
RMSE_test = mean_squared_error(y_test_pred, y_test, squared=False)
print(RMSE_test)

# Gradient Descent function
def GD(w0, alpha, epochs=500, X=None, y=None):
    m, n = X.shape
    for epoch in range(epochs):
        # Shuffle data at the start of each epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Stochastic Gradient Descent
        for i in range(m):
            rand_int = np.random.randint(0, m)
            x_i = X_shuffled[rand_int]
            y_i = y_shuffled[rand_int] 
            
            # Prediction
            prediction = np.dot(x_i, w0)
            error = prediction - y_i 
            
            # Gradient update
            grad = error * x_i
            w0 = w0 - alpha * grad 
    return w0

# Hyperparameters
alpha = 0.001
epochs = 1000

# Initialize weights (number of features)
w = np.ones(X_train_b.shape[1])

# Run Gradient Descent to get the final weights
weights = GD(w, alpha, epochs=epochs, X=X_train_b, y=y_train)

# Make predictions on the training and testing datasets
y_LMSE_train_pred = np.dot(X_train_b, weights)
y_LMSE_test_pred = np.dot(X_test_b, weights)

# Compute RMSE for training and testing datasets
LMS_RMSE_train = mean_squared_error(y_train, y_LMSE_train_pred, squared=False)
LMS_RMSE_test = mean_squared_error(y_test, y_LMSE_test_pred, squared=False)

# Output the results
print("LMS RMSE (Train):", LMS_RMSE_train)
print("LMS RMSE (Test):", LMS_RMSE_test)


# RMSE for train
LMS_RMSE_train = mean_squared_error(np.dot(X_train_b, w_0.T), y_train, squared=False)
print(LMS_RMSE_train)

# RMSE for test 
LMS_RMSE_test = mean_squared_error(np.dot(X_test_b, w_0.T), y_test, squared=False)
print(LMS_RMSE_test)
