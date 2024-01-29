# Car Purchase Amount Prediction

This Jupyter Notebook project aims to predict the amount a customer is likely to spend on a car purchase based on various features such as age, annual salary, credit score, and so on.

## Dataset

The dataset used in this project is located in the `csv` file. It contains the following columns:

- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Annual Salary`: Annual salary of the customer
- `Credit Score`: Credit score of the customer
- `Debt`: Amount of debt of the customer
- `Car Purchase Amount`: Amount spent by the customer on a car purchase (target variable)

## Notebook Files

- `car_purchase_amount_prediction.ipynb`: Jupyter Notebook containing the data exploration, preprocessing, model training, and evaluation.

## Model Used

This project utilizes two different approaches for prediction:

1. **Artificial Neural Networks (ANN)**:
   - An ANN model is trained using TensorFlow/Keras to predict the car purchase amount based on the input features.
   - The model architecture includes input, hidden, and output layers with appropriate activation functions.
   - Gradient descent algorithm is used to optimize the model parameters during training.

2. **Regression**:
   - A regression model is trained using scikit-learn to predict the car purchase amount based on the input features.
   - Various regression algorithms such as Linear Regression, Decision Tree Regression, or Random Forest Regression can be used for this purpose.

## Getting Started

To run this project locally, follow these steps:

1. Clone this repository to your local machine.
2. Open the Jupyter Notebook file `car_purchase_amount_prediction.ipynb` using Jupyter Notebook or JupyterLab.
3. Run the notebook cells sequentially to execute the code and see the results.

## Dependencies

This project requires the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras


