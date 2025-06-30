import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

train = pd.read_csv('C:/Users/LENOVO/Downloads/train.csv')
test = pd.read_csv('C:/Users/LENOVO/Downloads/test.csv')
gender_submission = pd.read_csv('C:/Users/LENOVO/Downloads/gender_submission.csv')

print("Train Data Info:")
print(train.info())

print("\nTest Data Info:")
print(test.info())

print("\nGender Submission Info:")
print(gender_submission.info())

print("\nTrain Summary Statistics:")
print(train.describe(include='all'))

print("\nMissing Values in Train Data:")
print(train.isnull().sum())

print("\nMissing Values in Test Data:")
print(test.isnull().sum())

sns.countplot(x='Survived', data=train)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Pclass', data=train)
plt.title('Passenger Class Distribution')
plt.show()sns.countplot(x='Sex', data=train)
plt.title('Gender Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(train['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(train['Fare'], kde=True)
plt.title('Fare Distribution')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Survival by Passenger Class')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Survival by Gender')
plt.show()

sns.countplot(x='Embarked', hue='Survived', data=train)
plt.title('Survival by Embarkation Port')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Survived', y='Age', data=train)
plt.title('Age vs Survival')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Survived', y='Fare', data=train)
plt.title('Fare vs Survival')
plt.show()

train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

test_with_predictions = pd.merge(test, gender_submission, on='PassengerId')
print("\nTest Data with Predictions:")
print(test_with_predictions.head())

summary = """
Summary of Titanic Dataset EDA:

- Total Passengers in Training Set: {}
- Survival Rate: {:.2f}%
- Higher survival rates observed in:
    - Females compared to males.
    - Passengers in 1st class.
    - Passengers who embarked from Cherbourg.
- Age distribution is right-skewed, most passengers are between 20-40 years.
- Higher fare seems to correlate with better survival chances.
""".format(len(train), train['Survived'].mean() * 100)

print(summary)

with open('Titanic_EDA_Summary.txt', 'w') as f:
    f.write(summary)
