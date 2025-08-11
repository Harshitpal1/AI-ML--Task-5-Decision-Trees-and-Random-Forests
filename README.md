# AI & ML Internship - Task 5: Decision Trees and Random Forests

This repository contains the solution for Task 5 of the AI & ML Internship. The project focuses on the use of Decision Trees and Random Forests for classification, using the Heart Disease dataset.

## Project Overview

The main goals of this project were to:
- Train and evaluate a Decision Tree classifier.
- Understand and control for overfitting in Decision Trees.
- Train and evaluate a Random Forest classifier.
- Compare the performance of the two models.
- Interpret the feature importances from the Random Forest.

## Results

- **Decision Tree Accuracy**: 98.54%
- **Random Forest Accuracy**: 98.54%
- **Random Forest Cross-Validation Accuracy**: 99.71%

The Random Forest model was found to be the most accurate and robust model for this dataset.

## Interview Questions

### 1. How does a decision tree work?
A decision tree is a type of supervised learning algorithm that is mostly used in classification problems. It works by splitting the data into subsets based on the value of the input features, creating a tree-like model of decisions.

### 2. What is entropy and information gain?
- **Entropy**: A measure of impurity or randomness in the data.
- **Information Gain**: The reduction in entropy achieved by splitting the data on a particular feature.

### 3. How is a random forest better than a single tree?
A Random Forest is an ensemble of many decision trees. It is better because it reduces overfitting, is more accurate, and more robust.

### 4. What is overfitting and how do you prevent it?
Overfitting is when a model learns the training data too well, including the noise, which leads to poor performance on new data. It can be prevented by pruning the tree, setting a maximum depth, or using a Random Forest.

### 5. What is bagging?
Bagging (Bootstrap Aggregating) is the technique of creating multiple models on random subsets of the training data and then aggregating their predictions.

### 6. How do you visualize a decision tree?
You can use `scikit-learn`'s `export_graphviz` function along with the `graphviz` library to create a visual representation of the tree.

### 7. How do you interpret feature importance?
Feature importance is a score that indicates how much a feature contributes to the model's accuracy. It is usually calculated as the average reduction in impurity caused by that feature.

### 8. What are the pros/cons of random forests?
- **Pros**: High accuracy, robust, less prone to overfitting.
- **Cons**: "Black box" model (hard to interpret), can be computationally expensive.
