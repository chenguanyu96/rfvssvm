# -*- coding: utf-8 -*-
import scipy.misc
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics
import constants
import utils

## BINARY CLASSIFICATION (RANDOM FOREST)
wine_data = utils.getdata('data/winequality-red.csv')
X = wine_data.drop(columns=['quality'])
y = pd.Series.to_frame(wine_data['quality'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=constants.TEST_SIZE, random_state=123)

y_test.loc[:,'poor'] = (y_test['quality'] <= constants.POOR_THRES_BIN)*1
y_test.loc[:,'good'] = (y_test['quality'] >= constants.GOOD_THRES_BIN)*1
y_train.loc[:,'poor'] = (y_train['quality'] <= constants.POOR_THRES_BIN)*1
y_train.loc[:,'good'] = (y_train['quality'] >= constants.GOOD_THRES_BIN)*1

rf_model = RandomForestClassifier(n_estimators=1000, criterion="entropy", 
                                  max_features=3, min_impurity_decrease=0.0, 
                                  min_samples_split=3, min_samples_leaf=2, 
                                  oob_score=True, random_state=constants.SEED, 
                                  class_weight=None, max_depth=None)
rf_model.fit(X_train, y_train['good'])
print("OOB Score (good): %.3f" % rf_model.oob_score_)
y_pred = rf_model.predict(X_test)
rf_model_metrics = metrics.classification_report(y_test['good'], y_pred)
print(rf_model_metrics)
confmat = metrics.confusion_matrix(y_test['good'], y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
twoclass_rf = ax.get_figure()
twoclass_rf.savefig(constants.FILE_DIR + "twoclass_rf.jpg")

loss_rf = (confmat[0][1]+confmat[1][0])/480.
print(loss_rf)
eps = np.sqrt((np.log(2./0.05))/(2.*1599.))
confint = [loss_rf-eps,loss_rf+eps]
print(confint)


## BINARY CLASSIFICATION (SUPPORT VECTOR MACHINES)
wine_data['quality_binary'] = pd.cut(wine_data['quality'], 
         bins=constants.BINARY_BINS, labels=constants.BGROUP_NAMES)
quality_binary = LabelEncoder()
wine_data['quality_binary'] = quality_binary.fit_transform(
        wine_data['quality_binary'])
X = wine_data.drop(columns=["quality"])
y = pd.Series.to_frame(wine_data['quality_binary'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=123)

svm_model = svm.SVC(kernel='poly')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
svm_model_metrics = metrics.classification_report(y_test, y_pred)
print(svm_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
twoclass_svm = ax.get_figure()
twoclass_svm.savefig(constants.FILE_DIR + "twoclass_svmpoly.jpg")

svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
svm_model_metrics = metrics.classification_report(y_test, y_pred)
print(svm_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
twoclass_svm = ax.get_figure()
twoclass_svm.savefig(constants.FILE_DIR + "twoclass_svmrbf.jpg")

loss_rbfsvm = (confmat[0][1]+confmat[1][0])/480.
print(loss_rbfsvm)
eps = np.sqrt((np.log(2./0.05))/(2.*1599.))
confint = [loss_rbfsvm-eps,loss_rbfsvm+eps]
print(confint)

x = ["Random Forest", "RBF SVM"]
y = [loss_rf, loss_rbfsvm]
error_graph = plt.errorbar(x, y, yerr=eps)
plt.xlabel("Algorithms")
plt.ylabel("True Risk / Error")
plt.savefig(constants.FILE_DIR + "binary_errorgraph.jpg")
