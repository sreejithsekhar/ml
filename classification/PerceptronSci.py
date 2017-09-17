import DataSet as ds
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import Plotter

X_train_std, X_test_std, y_train, y_test = ds.getIrisDataSet()

ppn = Perceptron(n_iter= 40, eta0= 0.1, random_state= 0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print("Misclassified samples: %d" % (y_test != y_pred).sum())
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt = Plotter.plot_decision_regions(X=X_combined_std, y=y_combined,
                                    classifier=ppn, test_idx=range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.show()
