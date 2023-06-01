# FraudDetection
Credit Card Fraud Detection.

<br>

## 1. Business Problem

The Bank wants to predict which transaction in fraudulent. 
A fraud is a strange transaction that can lead to data leakage. 

### 1.1 Data Information
The dataset ([https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]) contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation; the only features which have not been transformed with PCA are 'Time' and 'Amount'.


## 2. Solution Strategy

My solution to solve this problem will be the development of a data science project. This project will have a machine learning model which can predict used cars prices.

**Step 01. Exploratory Data Analysis:** The exploratory data analysis section consists of data description, search for missing values (not present), and correlation analysis. 

**Step 02. Data Preparation:** 
'Amount' variable is scaled using *StandardScaler*, while other explanatory variables are already transformed with PCA.
*PowerTransformer* is applied to every explanatory variable.
This dataset is highly imbalanced: only 0.17% of observation are fraudulent (class=1). Variables are transformed using SMOTE to perform better analysis. 
SMOTE is an over-sampling technique that synthesize new examples from the minority class.

**Step 03. Machine Learning Modeling:** Training of the machine learning algorithms and prediction of the data. StratifiedKFold cross validation is used for hyperparameters tuning and to reach better performances.  


## 3. Models comparison

#### Logistic Regression Model

|  ROCAUC score |    precision  |     recall    |    f1-score   |
|:-------------:|:-------------:|:-------------:|:-------------:|
|     0.986     |      0.53     |     0.94      |     0.55      |


#### Random Forest Classifier Model

|  ROCAUC score |    precision  |     recall    |    f1-score   |
|:-------------:|:-------------:|:-------------:|:-------------:|
|     0.981     |      0.62     |     0.93      |     0.69      |


#### Gradient Boosted Classifier Model

|  ROCAUC score |    precision  |     recall    |    f1-score   |
|:-------------:|:-------------:|:-------------:|:-------------:|
|     0.914     |      0.60     |     0.91      |     0.66      |

#### XGBoost Model

|  ROCAUC score |    precision  |     recall    |    f1-score   |
|:-------------:|:-------------:|:-------------:|:-------------:|
|     0.959     |      0.80     |     0.91      |     0.84      |



## 4. Conclusions

The chosen model was **XGBoost** and it was tuned to improve their parameters and scores.
The difference between models performace is small, so the decision was made considering that XGBoost is the fastest to train. 

Next steps would be to improve model performances checking more tuned parameters.
