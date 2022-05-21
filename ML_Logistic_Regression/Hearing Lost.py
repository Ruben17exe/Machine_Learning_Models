import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score, plot_roc_curve

scaler = StandardScaler()
model = LogisticRegression()

df = pd.read_csv("hearing_test.csv")
x = df[["age", "physical_score"]]
y = df["test_result"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=101)
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)
model.fit(x_train_scaler, y_train)
# y_pred = model.predict(x_test_scaler)
threshold = 0.58
y_pred = (model.predict_proba(x_test_scaler)[:, 1] > threshold).astype(int)
print(classification_report(y_test, y_pred))

# plot_roc_curve(model, x_test_scaler, y_test)
# plt.show()

threshold = []
accuracy = []
for p in np.unique(model.predict_proba(x_train_scaler)[:, 1]):
    threshold.append(p)
    y_pred = (model.predict_proba(x_train_scaler)[:, 1] >= p).astype(int)
    accuracy.append(balanced_accuracy_score(y_train, y_pred))

max_value_threshold = "Threshold => " + "Max Value: " + str(round(threshold[np.argmax(accuracy)], 5))
plt.plot(threshold, accuracy)
plt.xlabel(max_value_threshold)
plt.ylabel("Accuracy")
plt.show()
