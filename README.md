DASC41103 Project 2 – Group 23

🚀 Overview
This project applies a Multi-Layer Perceptron (MLP) neural network to predict whether an individual earns more than $50,000 per year using the UCI Adult Income dataset. The workflow includes:

Data preprocessing and feature engineering

Model building and hyperparameter tuning

Evaluation and reporting

Comparative analysis with classical machine learning models

📁 Repository Structure
text
Code/
│   └── MLP_Final.ipynb         # Main notebook with full analysis and implementation

Data/
│   ├── project_adult.csv       # Training and test dataset
│   └── project_validation.csv  # Additional/validation inputs

Output/
│   ├── results.csv             # Model predictions and evaluation metrics
│   └── figures/                # Visualization outputs

README.md                       # Project summary and analysis
📦 Folder Contents
Code/: Jupyter notebook containing data processing, model building, results, and analysis.

Data/: Raw census dataset and validation/test files.

Output/: Results, figures, saved metrics, and predictions.

README.md: This file, detailing project goals, workflow, and analysis responses.

📝 Analytical Questions & Answers
a. What is the objective of the project?
To predict whether a person's income is greater than $50,000 per year using an MLP model trained on demographic and occupational features from the UCI Adult Income dataset.

b. Who are the group members?
Jake Laurie

c. What is the dataset and which features are included?
Adult Income dataset from the UCI ML Repository.

Features: Age, workclass, education, marital status, occupation, relationship, race, sex, hours worked per week, native country.

Target: Income category (<=50K or >50K per year).

d. Which algorithm was used and how was it optimized?
Algorithm: Multi-Layer Perceptron (MLP) neural network, built with PyTorch/skorch.

Optimization:

Data preprocessing: Drop index columns, fill missing values, encode categoricals, scale numerics.

Hyperparameter tuning: Used GridSearchCV to optimize learning rate, batch size, optimizer (Adam/SGD), epochs, hidden layer structure, dropout, and activation functions (ReLU/Tanh/Sigmoid).

e. How was the model evaluated and what were the results?
Evaluation Metrics: Accuracy, precision, recall, F1-score (via classification report and confusion matrix).

Results:

Best test accuracy: ~85% using SGD optimizer, ReLU activation, and dropout regularization.

Hyperparameter tuning significantly impacted model performance.

Visualization and analysis revealed the importance of preprocessing and model configuration.

f. What are the key insights and limitations?
Insights:

MLPs can perform well on tabular data when highly optimized.

Proper feature scaling, encoding, and tuning are essential for neural network success.

Limitations:

Sensitive to hyperparameter settings and preprocessing.

May be outperformed by simpler models if not carefully tuned.

👥 Authors
[Add group member names here]
