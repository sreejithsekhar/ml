import DataSet as ds
import numpy as np
import Plotter as pt
from sklearn.neighbors import KNeighborsClassifier


X_train_std, X_test_std, y_train, y_test = ds.getIrisDataSet()
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
knn.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt = pt.plot_decision_regions(X_combined_std, y_combined, classifier=knn,
                         test_idx=range(105, 150))

plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.show()
