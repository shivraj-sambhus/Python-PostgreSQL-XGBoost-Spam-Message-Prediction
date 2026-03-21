import pandas as pd
import sqlalchemy 
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#EXTRACT PHASE START

#In this section, I'll extract the dataset and understand its features. The dataset consists of a label (spam or ham) and a message but it needs to be separated by a tab character.

sms_df = pd.read_csv("SMSSpamCollection", sep= "\t", names=["label", "message"])

#print(sms_df.info())

#print(sms_df)

#EXTRACT PHASE END
#TRANSFORM PHASE START

# In this section, I'll perform data cleaning and validation and add some essential columns to the SMS dataframe.

# Drop any null and duplicate values if there are any. 

sms_df = sms_df.dropna()

sms_df = sms_df.drop_duplicates()

# Remove any leading and trailing characters and whitespace. This is essential for calculating word and character counts.
sms_df["message"] = sms_df["message"].str.strip()

# Remove any double spaces.
sms_df["message"] = sms_df["message"].str.replace(r"\s+", " ", regex=True)

# Reset the index to avoid gaps.

sms_df = sms_df.reset_index(drop=True)

#print(sms_df)

# Create an identifier for each message

sms_id = range(1, len(sms_df) + 1)
sms_df.insert(0,'sms_id',sms_id)

# Add a numeric column to indicate that a message is spam or not

numeric_label = []

for label in sms_df["label"]:
    if label == "spam":
        numeric_label.append(1)
    else:
        numeric_label.append(0)

sms_df.insert(1, 'numeric_label', numeric_label)

#sms_df["numeric_label"] = numeric_label

# Add a column to indicate the character count of each message.
   
character_count = []

for sms in sms_df["message"]:
    count = len(sms)
    character_count.append(count)

sms_df["character_count"] = character_count

#print(sms_df)

# Add a column to indicate the word count of each message. The 'words' are separated by a space.
word_count = []

for sms in sms_df["message"]:
    words = sms.split()
    count = len(words)
    word_count.append(count)

sms_df["word_count"] = word_count

#print(sms_df)

# Add a column to indicate the number of spam keywords in each message. I'll need to first create a list of words that are assumed to be spam. I'll make the list as short as possible in order to exclude non-spam (ham) words.

spam_list = ["prize!", "prize", "winner", "win", "urgent!", "free", "congrats!", "money!!!", "sexy!!", "sexy", "sex", "subscription", "subscribed", "charged", "horny", "naked"]

spam_word_count = []

for sms in sms_df["message"]:
    sms_lower = sms.lower()
    count = 0
    for word in spam_list:
        count += sms_lower.count(word)
    spam_word_count.append(count)

sms_df["spam_word_count"] = spam_word_count

#print(sms_df)

# Create a column to indicate if a message has a hyperlink. This will help me predict if the message is spam.

has_link = []

for sms in sms_df["message"]:
    if ("https://" in sms) or ("http://" in sms) or ("www." in sms):
        has_link.append(1)
    else:
        has_link.append(0)

sms_df["has_link"] = has_link

#print(sms_df)

#TRANSFORM PHASE END
#LOAD PHASE START


engine = create_engine(
    "postgresql://postgres:157240@localhost:5432/ETL_XGBoost"
)

with engine.begin() as conn:
    conn.execute(text('DELETE FROM "SMS";'))  # Clear old rows

sms_df.to_sql("SMS", engine, if_exists="append", index=False)

#LOAD PHASE END
#XGBOOST PHASE START

# In PostgreSQL, I created indexes on the numeric_label and word_count, a view that displays spam sms summary statistics, and a materialized view that displays the numeric columns in the spam_df dataframe.
# Let's load in the materialized view.

query = "SELECT * FROM sms_training_data"

sms_training_df = pd.read_sql(query, engine)

print(sms_training_df)

# Creating the XGBoost model, including testing and training and TF-IDF text features.

vectorizer = TfidfVectorizer(stop_words="english")

X_text = vectorizer.fit_transform(sms_df["message"])


X_numeric = sms_training_df[[
    "word_count",
    "character_count",
    "spam_word_count",
    "has_link"
]]

X = hstack([X_text, X_numeric])
y = sms_training_df["numeric_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# The XGBoost model with the TF-IDF features is 97.2%
#print("Accuracy:", accuracy_score(y_test, y_pred))

# Next, we print the classification report

#print(classification_report(y_test, y_pred))

# Next, we print the feature importance

importances = model.feature_importances_

#plt.bar(range(len(importances)), importances)
#plt.title("Feature Importance")
#plt.show()

# Next, we print the ROC curve

y_probs = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
plt.plot([0,1], [0,1], linestyle="--")

font1 = {'family':'serif','color':'black','size':20}

plt.xlabel("False Positive Rate", fontdict=font1)
plt.ylabel("True Positive Rate", fontdict=font1)
plt.title("ROC Curve – SMS Spam Classifier", fontdict=font1)

plt.legend(loc="lower right")
plt.grid()

plt.show()

# Compare the results with two other models: Logistic Regression and Decision Tree

# Fit the logistic regression model and print the classification report

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_probs_log = log_model.predict_proba(X_test)[:, 1]
y_pred_log = log_model.predict(X_test)

#print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

# Fit the decision tree model and print the classification report

dt_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_probs = dt_model.predict_proba(X_test)[:,1]
y_pred_dt = dt_model.predict(X_test)

# Compute the classification report for model comparison

print("\nLogistic Regression Classification Report")
print("----------------------------------------")
print(classification_report(y_test, y_pred_log))

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

print("\nModel Comparison")
print("-------------------")
print("Logistic Regression:", accuracy_score(y_test, y_pred_log))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("XGBoost:", accuracy_score(y_test, y_pred))

print("\nDecision Tree Classification Report")
print("----------------------------------")
print(classification_report(y_test, y_pred_dt))

print("\nXGBoost Classification Report")
print("-----------------------------")
print(classification_report(y_test, y_pred))


# Compute the ROC-AUC for model comparison


print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_probs_log))
print("Decision Tree AUC:", roc_auc_score(y_test, dt_probs))
print("XGBoost AUC:", roc_auc_score(y_test, y_probs))