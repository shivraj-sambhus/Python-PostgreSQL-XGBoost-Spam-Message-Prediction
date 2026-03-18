# Python-PostgreSQL-XGBoost-Spam-Message-Prediction

![A receiver operating characteristic (ROC) curve that illustrates the accuracy of the XGBoost model used to predict whether an SMS is spam.](ROC_Curve.png)

A receiver operating characteristic (ROC) curve that illustrates the accuracy of the XGBoost model used to predict whether an SMS is spam.

# 1. Objective

I want to build a predictive model that classifies a text message (SMS) as spam or not spam. In this project, I will use an unstructured dataset from 2012 found on the UC Irvine machine learning repository that contains SMS data from students of the National University of Singapore. Since the data is unstructured, I will need to first extract meaningful features from it and create a structured version of it. I will do this by creating an ETL pipeline and loading it in PostgreSQL to generate a materialized view used in the predictive model. To select the final model, I will use an iterative approach involving logistic regression, decision trees, and an XGBoost model.

Link to the data: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

# 2. Requirements
```
Python 3.13
pandas
sqlalchemy
scikit-learn
matplotlib
xgboost
PostgreSQL 18.3
```
# 3. ETL Process

Let's first take a look at the data. All we can really see is each message and its corresponding label. The messages seem full of typos and punctuation errors, typical of spam data.

```
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham	U dun say so early hor... U c already then say...
```

First, I'll extract the data using Pandas, and create two columns: label and message. There are 5572 rows and 2 columns. 

Next, I'll transform the data by cleaning it and generating features to be used in the predictive model. For data cleaning, I'll drop empty rows and duplicates and also remove whitespace and trailing characters. This will be useful when creating features in the dataset like character count and word count. For feature creation, I'll use simple FOR loops to iterate through each message and create a numeric label for spam/ham, character count, word count, spam word count, and a label to indicate the presence of hyperlinks in the message.

Next, I'll load the data to PostgreSQL by using the sqlalchemy library.

# 4. PostgreSQL Integration

# 5. Iterative Approach

# 5.1 Logistic Regression

# 5.2 Decision Tree

# 5.3 XGBoost Model

# 6. Insights





