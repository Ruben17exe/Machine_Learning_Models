import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

df = pd.read_csv("medical_report.csv")
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'Skin', 'Insulin', 'BMI', 'Pedigree', 'Age']
x = df[feature_cols]
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

d_tree = DecisionTreeClassifier()
d_tree.fit(x_train, y_train)
y_pred = d_tree.predict(x_test)
print(classification_report(y_test, y_pred))

dot_data = StringIO()
export_graphviz(d_tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Image.png')
Image(graph.create_png())
