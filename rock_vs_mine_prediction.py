
Importing the Dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection and Data Processing"""

#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('/content/sonar data.csv', header=None)

sonar_data.head()

# number of rows and columns
sonar_data.shape

sonar_data.describe()  #describe --> statistical measures of the data

sonar_data[60].value_counts()

"""M --> Mine

R --> Rock
"""

sonar_data.groupby(60).mean()

# separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

print(X_train)

print(Y_train)

"""Model Training --> Logistic Regression"""

model = LogisticRegression()

#training the Logistic Regression model with training data
model.fit(X_train, Y_train)

"""Model Evaluation"""

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data : ', test_data_accuracy)

"""Making a Predictive System"""

input_data = (0.0116,0.0744,0.0367,0.0225,0.0076,0.0545,0.1110,0.1069,0.1708,0.2271,0.3171,0.2882,0.2657,0.2307,0.1889,0.1791,0.2298,0.3715,0.6223,0.7260,0.7934,0.8045,0.8067,0.9173,0.9327,0.9562,1.0000,0.9818,0.8684,0.6381,0.3997,0.3242,0.2835,0.2413,0.2321,0.1260,0.0693,0.0701,0.1439,0.1475,0.0438,0.0469,0.1476,0.1742,0.1555,0.1651,0.1181,0.0720,0.0321,0.0056,0.0202,0.0141,0.0103,0.0100,0.0034,0.0026,0.0037,0.0044,0.0057,0.0035)
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')

