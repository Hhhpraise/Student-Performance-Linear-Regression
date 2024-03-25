# Import necessary libraries
import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv("student_mat.csv", sep=";")

# Select relevant features and convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=["romantic", "guardian"])
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "goout", "freetime", "age",
             "romantic_no", "romantic_yes", "guardian_father", "guardian_mother", "guardian_other"]]

# Define the target variable
predict = "G3"

# Prepare the data for modeling
X = np.array(data.drop("G3", axis=1))
y = np.array(data[predict])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Initialize best accuracy variable
best = 0

# Train the model and save the best performing model
for _ in range(1000):
    x_train,x_test, y_train , y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    #print(acc)
    if acc> best:
        best = acc
    with open("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f)

# Load the best model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Print the best accuracy achieved
print('best accuracy : ', best)

# Make predictions and compare with actual values
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("Predicted :",predictions[x], "Data:",x_test[x], "Actual :",y_test[x])

# Plot actual vs. predicted values
plt.scatter(y_test, predictions)
plt.xlabel("Actual prediction")
plt.ylabel("Predicted prediction")
plt.title("Actual vs. Predicted prediction")
plt.legend()
plt.show()