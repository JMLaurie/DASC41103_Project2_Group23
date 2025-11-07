DASC41103 Project 2 – Group 23
🚀 Overview
This project applies a Multi-Layer Perceptron (MLP) neural network to predict whether an individual earns more than $50,000 per year using the UCI Adult Income dataset. The workflow includes:

Data preprocessing and feature engineering

Model building and hyperparameter tuning

Evaluation and reporting

Comparative analysis with classical machine learning models

📁 Repository Structure
Code/: Contains MLP_Final.ipynb – main notebook with full analysis and implementation

Data/: Census dataset (project_adult.csv) and validation inputs

Output/: Model predictions, evaluation metrics, and figures

README.md: Project summary and analysis

📝 Analytical Questions & Answers
a. Why did you choose the specific architecture (e.g., number of layers, activation functions) for each model?

The architecture was chosen based on systematic grid search and cross-validation to balance complexity and generalization. For tabular data like the Adult Income dataset, we used a modest number of hidden layers (typically 1-2) and units to reduce risk of overfitting, following practical recommendations and benchmarks in neural network literature. ReLU was the default activation due to its effectiveness and faster convergence in deep learning tasks, but Tanh/Sigmoid were tested for comparison. Dropout was added for regularization, and SGD/Adam optimizers were evaluated for training stability. Layer depth, activation, and regularization were all tuned using validation set performance and comparative analysis with classical models.​

b. How did you monitor and mitigate overfitting in your models?

Overfitting was monitored by tracking validation accuracy/loss during training. Techniques used:

Dropout: Randomly deactivates neurons during training to prevent reliance on specific weights.

Early Stopping: Training was stopped if validation loss increased for several epochs.

Regularization: L2 weight penalty was applied.

Data preprocessing and feature engineering: Ensured clean, scaled inputs and encoded categorical variables to improve generalization.

Hyperparameter tuning: Used cross-validation and GridSearchCV to identify model settings that yielded robust validation performance but did not inflate train metrics.​

c. What ethical concerns might arise from deploying models trained on these datasets?

The Adult Income dataset contains sensitive demographic features, such as gender, race, and age, which can reflect or perpetuate real-world biases if used improperly. Ethical concerns include:

Bias amplification: Models may inherit and magnify historical inequalities, resulting in unfair income predictions for certain groups.

Discrimination risk: Automated decision-making using biased data can reinforce societal injustices.

Privacy: Potential for misuse of sensitive demographic data.

Responsible deployment should include fairness analysis, bias mitigation steps (such as fairness-aware training), and transparency about limitations.​

d. Why are activation functions necessary in neural networks?

Activation functions are essential because they introduce non-linearity, allowing neural networks to model complex relationships and patterns in the data. Without them, the entire network reduces to a linear transformation, which cannot capture the intricacies required in tasks such as classification or regression. Functions like ReLU, Tanh, and Sigmoid allow networks to approximate highly complex, non-linear systems, enabling successful deep learning on real-world datasets.​

👥 Authors
Jake Laurie and Jordan Short
