# Student-Performance-Linear-Regression
This repository contains code for a student performance prediction model implemented using linear regression. The model utilizes relevant features from a student dataset and employs one-hot encoding to handle categorical variables. Additionally, the code includes data preprocessing, model training, evaluation, and visualization of predictions using matplotlib.

## How the Code Works
1. **Data Loading and Preprocessing**
- The code begins by loading the student dataset from a CSV file using pandas.
- Relevant features are selected for the prediction model, and categorical variables are converted to numerical format using one-hot encoding.
- The final dataset includes features such as first period grade (G1), second period grade (G2), study time, failures, absences, social time (go out), free time, age, as well as one-hot encoded variables for romantic status and guardian.
2. **Model Training**
- The target variable for prediction is set as the final period grade (G3).
- The data is split into input features (X) and the target variable (y) for model training.
- The training data is further split into training and testing sets using a 90/10 split.
- Linear regression model is instantiated and trained on the training data.
3. **Model Evaluation and Selection**
- The model is evaluated for its accuracy on the testing data and the best performing model is saved for future use.
- The process is repeated for 1000 iterations, and the best accuracy achieved is printed at the end of the training process.
4. **Model Prediction**
- The best performing model is loaded from the saved pickle file.
- Predictions are made using the testing data, and the predicted values are compared with the actual values.
5. **Visualization**
- A scatter plot is created to visualize the relationship between actual and predicted grades using matplotlib.
- The plot provides a visual representation of how well the model's predictions align with the actual values.

## Running the Code
- Ensure that Python and the required packages (pandas, numpy, sklearn, matplotlib) are installed.
- Download the "student_mat.csv" dataset or use your own dataset with similar structure.
- Execute the provided Python script in an environment with the necessary packages installed.

## File Structure
- student_mat.csv: Sample dataset used for the model (You can replace with your own dataset)
- studentflow.py: Python script containing the entire process for student performance prediction using linear regression.
- studentmodel.pickle: Saved best performing model for future use.

## Credits
- This project is inspired by the work of various contributors in the fields of machine learning, data analysis, and education research.
- Feel free to explore, use, and modify the code as needed. If you have any questions or suggestions, please feel free to reach out and contribute to this project. Thank you for your interest in the student performance prediction model!
