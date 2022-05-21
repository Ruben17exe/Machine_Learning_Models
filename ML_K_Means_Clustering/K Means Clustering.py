import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()
columns = iris.feature_names
samples = iris.data
target = iris.target
target_names = iris.target_names

model = KMeans(n_clusters=3)
model.fit(samples)

new_samples = np.array([[5.7, 4.4, 1.5, 0.4],
                        [6.5, 3., 5.5, 0.4],
                        [5.8, 2.7, 5.1, 1.9]])

predicted_names = [target_names[index] for index in model.predict(new_samples)]
print("Predicted Values:", ", ".join(predicted_names))

# Inertia depending on the number of cluster
num_clusters = [1, 2, 3, 4, 5, 6, 7, 8]
inertias = []

for k in num_clusters:
  model_ev = KMeans(n_clusters=k)
  model_ev.fit(samples)
  inertias.append(model_ev.inertia_)

plt.plot(num_clusters, inertias, '-o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()
