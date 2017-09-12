from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np

# We load the data with load_iris from sklearn
data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']

#Normalization
#Subtract the mean for each feature
features -= np.mean(features, axis=0)
#Divide each feature by its standard deviation
features /= np.std(features, axis=0)

#Binary label
is_versicolor = target == 1
binary_target = np.zeros(len(target))
binary_target[is_versicolor] = 1

#5 Fold Cross Validation
kf = KFold(n=len(binary_target), n_folds=5, shuffle=True)

cv = 0
for tr, tst in kf:

    #Train Test Split
    tr_features = features[tr, :]
    tr_target = binary_target[tr]

    tst_features = features[tst, :]
    tst_target = binary_target[tst]

    #Training Logistic Regression
    # model = LogisticRegression()
    # model.fit(tr_features, tr_target)

    #Training SVM Model
    model = SVC()
    model.fit(tr_features, tr_target)

    #Measuring training and test accuracy
    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)

    print "%d Fold Train Accuracy:%f, Test Accuracy:%f" % (
        cv, tr_accuracy, tst_accuracy)
    cv += 1







