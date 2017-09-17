import DataSet as ds
import numpy as np
from sklearn.svm import SVC
import Plotter as pt


X_train_std, X_test_std, y_train, y_test = ds.getIrisDataSet()
svm = SVC(kernel="linear", C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt = pt.plot_decision_regions(X_combined_std, y_combined, classifier=svm,
                         test_idx=range(105, 150))

plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.show()

