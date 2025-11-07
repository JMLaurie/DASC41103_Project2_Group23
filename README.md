## ***DASC41103 Project 2 – Group 23***

***Overview:***  
This project applies a Multi-Layer Perceptron (MLP) neural network to predict whether an individual earns more than $50,000 per year using the UCI Adult Income dataset. The workflow includes:

- Data preprocessing and feature engineering
- Model building and hyperparameter tuning
- Evaluation and reporting
- Comparative analysis with classical machine learning models

## ***Repository Structure***
- `Code/`: Contains `MLP_Final.ipynb` – main notebook with full analysis and implementation
- `Data/`: Census dataset (`project_adult.csv`) and validation inputs
- `Output/`: Model predictions, evaluation metrics, and figures
- `README.md`: Project summary and analysis

## ***Analytical Questions & Answers***

### **a. Why did you choose the specific architecture (e.g., number of layers, activation functions) for each model?**
***
The architecture was chosen based on grid search and cross-validation to balance complexity and generalization. For tabular data like the Adult Income dataset, a modest number of hidden layers (1-2) and units was used to reduce overfitting. ***ReLU*** activation was standard because it helps training converge faster; ***Tanh*** and ***Sigmoid*** were tested for comparison. Dropout and regularizers were used to further minimize overfit. Parameter choices were validated against performance metrics and benchmarks from published neural network literature.

### **b. How did you monitor and mitigate overfitting in your models?**
***
***Overfitting was monitored*** by tracking validation accuracy/loss. Mitigation techniques included:
- ***Dropout:*** Randomly disables neurons to generalize learning.
- ***Early Stopping:*** Stopped training when validation loss stopped improving.
- ***Regularization:*** L2 penalties were applied to weights.
- ***Feature engineering:*** Categorical variables were encoded and scaled.
- ***Hyperparameter tuning:*** Used grid search, cross-validation, and comparison to test models for robustness.

### **c. What ethical concerns might arise from deploying models trained on these datasets?**
***
- ***Bias amplification:*** Sensitive features (gender, race) can reflect and reinforce historical biases.
- ***Discrimination:*** Automated predictions may unfairly impact specific demographic groups.
- ***Privacy:*** Use of personal demographic features raises data privacy concerns.
- ***Responsible deployment requires fairness analysis, bias checks, transparency, and proper handling of demographic data.***

### **d. Why are activation functions necessary in neural networks?**
***
***Activation functions introduce non-linearity,*** allowing neural networks to model complex and nonlinear relationships. Without them, layers would just perform linear transformations, preventing the network from learning meaningful or complex patterns. Functions such as ***ReLU, Tanh, and Sigmoid*** enable networks to approximate real-world, non-linear problems.

## ***Authors***
Jake Laurie and Jordan Short

