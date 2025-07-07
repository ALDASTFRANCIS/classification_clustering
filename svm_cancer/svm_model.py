import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Breast Cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred))

# User Input
print("\nEnter Feature Data:")
data_input = [float(input(f"{col}: ")) for col in data.feature_names[:5]]
data_input_scaled = scaler.transform([data_input + [0]*(len(data.feature_names)-5)])
pred = model.predict(data_input_scaled)[0]
print("Prediction:", "Malignant" if pred == 0 else "Benign")


# 14.71,21.59,95.55,656.9,0.1137,0.1365,0.1293,0.08123,0.2027,0.06758,0.4226,1.15,2.735,40.09,0.003659,0.02855,0.02572,0.01272,0.01817,0.004108,17.87,30.7,115.7,985.5,0.1368,0.429,0.3587,0.1834,0.3698,0.1094,0