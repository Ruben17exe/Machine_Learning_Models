from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris_data = load_iris()
# print(iris_data.data[0])
# print(iris_data.feature_names)
# print(iris_data.target)
# print(iris_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(iris_data.data,
                                                                                      iris_data.target,
                                                                                      test_size=0.2,
                                                                                      random_state=100)

k_list = [i for i in range(1, 101)]
accuracies = []
for k in k_list:  
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel("K Nearest Neighbors")
plt.ylabel("Validation Accuracy")
plt.title("Iris Classifier Accuracy")
plt.show()
