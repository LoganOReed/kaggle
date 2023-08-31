# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split

# "Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"
# All Features Given

# Setup Data
train_data = pd.read_csv("data/titanic/train.csv")
test_data = pd.read_csv("data/titanic/test.csv")

y_train = train_data["Survived"]
X_train = train_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
X_train = pd.get_dummies(X_train, columns=["Sex", "Embarked"], dtype=float)
mean_age = X_train["Age"].mean()
X_train["Age"].replace(np.nan, mean_age, inplace=True)
# Print the number of nulls in each column
# print(X_train.isnull().sum())
# "Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"

X_test = test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked"], dtype=float)
mean_age = X_test["Age"].mean()
X_test["Age"].replace(np.nan, mean_age, inplace=True)
X_test["Fare"].replace(np.nan, 0.0, inplace=True)
# Print the number of nulls in each column
# print(X_test.isnull().sum())

# print(X_test.head())

# Setup Model
M = RandomForestClassifier(max_depth=5, random_state=1)
M.fit(X_train, y_train)

pred = M.predict(X_test)
output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": pred})
output.to_csv("data/titanic/submission.csv", index=False)
# print(output)
# 0.778 Accuracy on Kaggle
