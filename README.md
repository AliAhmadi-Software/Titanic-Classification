
# Titanic Classification - Machine Learning in Python

## Project Overview

The goal is to build a machine learning model that predicts the survival of passengers on the Titanic based on various attributes like age, gender, class, etc.
The dataset used in this project is sourced from Kaggle's Titanic dataset.

### Key Objectives
- Preprocess the data to handle missing values, convert categorical data, and scale features.
- Explore data with visualizations to understand patterns and relationships.
- Train multiple machine learning models to predict survival outcomes.
- Evaluate and compare model performance using accuracy and other relevant metrics.

## Project Structure and Code Sections

The notebook is organized into multiple sections:
1. **Data Loading and Preprocessing**: Handling missing values, encoding categorical features, and data scaling.
2. **Exploratory Data Analysis (EDA)**: Data visualization to find patterns related to survival.
3. **Model Training**: Training different models including Logistic Regression, Decision Trees, Random Forest, etc.
4. **Model Evaluation**: Assessing model performance on test data.
5. **Final Results**: Analyzing results and choosing the best-performing model.

## Installation

To run this notebook, you need the following dependencies:

- Python 3.7+
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook (or Jupyter Lab)

Install the dependencies by running:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Running the Notebook

1. Clone the repository or download the notebook file.
2. Open the notebook in Jupyter Notebook or Jupyter Lab.
3. Execute the cells in order to reproduce the analysis and model results.

## Model Performance Summary

The following table summarizes the performance of different models used in this project based on accuracy, precision, recall, and F1 score.

| Metric       | Logistic Regression | KNN Classifier | Decision Tree | Random Forest | SVM   | XGBoost |
|--------------|---------------------|----------------|---------------|---------------|-------|---------|
| Accuracy     | 80.45%              | 80.45%        | 77.09%        | 81.56%        | 82.12%| 82.68%  |
| Precision    | 78.26%              | 78.26%        | 72.00%        | 78.87%        | 83.87%| 79.45%  |
| Recall       | 72.97%              | 72.97%        | 72.97%        | 75.68%        | 70.27%| 78.38%  |
| F1 Score     | 75.52%              | 75.52%        | 72.48%        | 77.24%        | 76.47%| 78.91%  |

### Conclusion
From the results, **SVM** and **XGBoost** performed the best in terms of accuracy, with XGBoost also achieving a high F1 score, making it a suitable choice for this classification task.