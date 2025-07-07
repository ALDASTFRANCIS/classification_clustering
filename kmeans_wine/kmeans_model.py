import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(df)

print("Cluster Centers:\n", model.cluster_centers_)

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis')
plt.title("K-Means Clustering on Wine Dataset")
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])
plt.show()

print("\nEnter Wine Data:")
features = wine.feature_names
data = [float(input(f"{f}: ")) for f in features]
pred_cluster = model.predict([data])[0]
print(f"Predicted Cluster: {pred_cluster}")


# 13.17,2.59,2.37,20.0,120.0,1.65,0.68,0.53,1.46,9.3,0.6,1.62,840.0,2