from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
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

#Training Logistic Regression
lr = LogisticRegression()
lr.fit(features, binary_target)

#Measuring accuracy
accuracy = np.mean(lr.predict(features) == binary_target)
print "Training Accuracy: %f" % accuracy
