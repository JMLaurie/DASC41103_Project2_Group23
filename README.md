DASC41103 Project 2 — Group 23
🚀 Project Objective
Predict whether an individual's income exceeds $50K/year using neural networks and census data.
This project showcases the effectiveness of a Multi-Layer Perceptron (MLP) for tabular classification and explores how data processing and hyperparameter tuning impact model performance.

👥 Group Members
[Add members' names here]

📊 Dataset Description
Source: UCI Machine Learning Repository — Adult (Census Income) dataset

Features: Age, education, occupation, marital status, and more

Target: Income category (<=50K or >50K per year)

🤖 Algorithm & Methods
Model: Multi-Layer Perceptron (MLP) with PyTorch and skorch

Preprocessing:

Dropped unnamed/index columns

Imputed missing values (mean/mode)

One-Hot Encoding for categoricals, Standard Scaling for numerics

Label Encoding for the target

Training: Stratified 80/20 train-test split

Hyperparameter Tuning: Used GridSearchCV for:

Learning rate

Batch size

Optimizer (Adam, SGD)

Number of epochs

Hidden layer sizes & number of layers

Dropout rate

Activation function (ReLU, Tanh, Sigmoid)

Evaluation: Cross-entropy loss, accuracy, precision, recall, F1-score

📈 Model Evaluation
Best Accuracy: ~85% (stratified cross-validation)

Key Metrics: Accuracy, precision, recall, F1-score on test set

Findings: Optimal results with ReLU activation, dropout, and SGD optimizer. Tuning had a significant effect.

📝 Results Summary
MLP matches classical ML models on tabular data when optimized

Best architecture: ReLU + SGD + Dropout

Importance of preprocessing and tuning: Critical for deep learning success

Limitations: Sensitive to feature scaling and hyperparameters

🛠️ How To Run
Clone the repo

Install requirements:
pip install torch sklearn skorch pandas numpy

Place data files in /Data

Open and run MLP_Final.ipynb

📁 Repository Structure
/Code: Source code and Jupyter notebooks

/Data: Dataset files

/Output: Model results, figures
