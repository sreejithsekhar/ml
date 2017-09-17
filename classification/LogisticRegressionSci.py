from sklearn.linear_model import LogisticRegression
import DataSet as ds
import Plotter as pt
import numpy as np
from sklearn.metrics import accuracy_score

X_train_std, X_test_std, y_train, y_test = ds.getIrisDataSet()

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)

print("Misclassified samples: %d" % (y_test != y_pred).sum())
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt = pt.plot_decision_regions(X=X_combined_std, y=y_combined,
                               classifier=lr, test_idx=range(105, 150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.show()
