# Model Card

## Model Details
This model is a Random Forest Classifier implemented using the Scikit-learn library. It is designed to predict whether an individual's income exceeds $50K per year based on census data. The model uses both categorical and numerical features as input. The hyperparameters for the Random Forest Classifier include n_estimators=100 and random_state=42.

The input features include:
- **Categorical features**: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Numerical features**: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week

## Intended Use
This model is designed to demonstrate how to build a machine learning pipeline for binary classification tasks. While it is primarily intended for educational purposes, it could be adapted for real-world applications with additional validation and testing. The model works with preprocessed census data and assumes that the input data follows the same format and preprocessing steps as the training data.

## Training Data
The model was trained on the Census Income Dataset, which contains demographic and income information. The dataset includes 32,561 rows and 15 columns. The data was split into training and test sets, with 80% of the data used for training and 20% reserved for testing. The training data was preprocessed using one-hot encoding for categorical features and label binarization for the target variable.

## Evaluation Data
The test set, which consists of 20% of the original dataset, was used to evaluate the model's performance. The test data was preprocessed using the same pipeline as the training data to ensure consistency. The evaluation metrics were computed on this test set.

## Metrics
The model's performance was evaluated using the following metrics:
- **Precision**: 0.7419
- **Recall**: 0.6384
- **F1 Score**: 0.6863

These metrics were computed using the `compute_model_metrics` function, which calculates precision, recall, and F1 score based on the model's predictions and the true labels in the test set.

## Ethical Considerations
The dataset used for training and evaluation may contain biases related to sensitive attributes such as race, gender, or marital status. These biases could influence the model's predictions and lead to unfair outcomes. It is important to evaluate the model's performance on different slices of the data (e.g., based on race or sex) to identify and address potential biases.

Additionally, the model's predictions are based on historical data, which may not reflect current societal or economic conditions. Care should be taken to ensure that the model is not used in contexts where fairness and equity are critical without thorough validation.

## Caveats and Recommendations
The model has several limitations:
1. The predictions are based on historical data and may not generalize well to new or unseen data.
2. The model assumes that the input data is preprocessed in the same way as the training data. Any deviation in preprocessing could lead to inaccurate predictions.
3. The model's performance on different slices of the data should be evaluated to ensure fairness and identify potential biases.

It is recommended to:
- Perform further hyperparameter tuning or experiment with alternative models (e.g., Gradient Boosting) to improve performance.
- Regularly evaluate the model's performance on updated datasets to ensure its relevance and accuracy.
- Use the model only for educational purposes or as a baseline for further development.