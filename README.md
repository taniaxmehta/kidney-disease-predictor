# Chronic Kidney Disease Prediction
Chronic Kidney Disease (CKD) is a medical condition characterized by the progressive loss of kidney function, impairing their ability to filter blood effectively. It poses significant complications and affects approximately 10% of Indians, with millions more at heightened risk. The dataset is tailored to the Indian population, emphasizing the importance of early detection for potentially life-saving interventions. Timely intervention can prevent the progression of kidney disease to kidney failure, underscoring the critical need for proactive healthcare measures.

## Exploratory Data Analysis
* CKD Cases Count: Visualizing the distribution of CKD cases provides a fundamental understanding of the dataset's class balance, setting the stage for subsequent analysis.
* Missing Values: An exploration of missing values column-wise offers insights into data quality and potential areas for data imputation or cleaning.
* Category Graph Observations: Graphs and histograms to uncover patterns within categorical and continuous variables, highlighting potential relationships
* Correlation Matrix: Provides a comprehensive view of the relationships between variables, guiding feature selection and engineering


## Data Pre-Processing
* Standardization: With StandardScaler, we ensure that each feature contributes equally to model training without being influenced by different scales
* Categorical Transformation: One-hot encoding enables the model to comprehend categorical data while preserving the integrity of the original features

## Logistic Regression
* Train-Test Split: Prevents overfitting, we split the dataset into training and testing subsets, ensuring the model's ability to generalize to unseen data
* Logistic Regression Model: With LogisticRegression class to construct a logistic regression model, enabling us to model the probability of CKD presence based on input attributes

## Model Evaluation
* Classification Report: A comprehensive overview of the model's performance, encompassing metrics such as precision, recall, F1-score, and support for each class
* Confusion Matrix: Visual representations of confusion matrices provide a clear depiction of the model's predictive accuracy and error rates, aiding in the interpretation of results





