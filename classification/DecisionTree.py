from sklearn.tree import DecisionTreeClassifier
import DataSet as ds
import numpy as np
from sklearn.svm import SVC
import Plotter as pt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
tree.fit(X_train, y_train)

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plt = pt.plot_decision_regions(X_combined_std, y_combined, classifier=tree,
                         test_idx=range(105, 150))

plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.show()

export_graphviz(tree, out_file="tree.dot", feature_names=["petal length", "petal width"])
