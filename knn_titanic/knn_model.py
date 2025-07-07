import os, sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = pd.read_csv("data/titanic.csv")
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred))

# User Input
features = ['Pclass', 'Sex (0=female, 1=male)', 'Age', 'SibSp', 'Parch', 'Fare']
print("\nEnter Passenger Data:")
data = [float(input(f"{f}: ")) for f in features]
user_pred = model.predict([data])[0]
print("Prediction:", "Survived" if user_pred == 1 else "Did not survive")
