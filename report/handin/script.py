# -*- coding: utf-8 -*-
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
wine_data['quality_binary'] = pd.cut(wine_data['quality'], 
         bins=constants.BINARY_BINS, labels=constants.BGROUP_NAMES)
quality_binary = LabelEncoder()
wine_data['quality_binary'] = quality_binary.fit_transform(
        wine_data['quality_binary'])
X = wine_data.drop(columns=["quality"])
y = pd.Series.to_frame(wine_data['quality_binary'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=123)

rf_model = RandomForestClassifier(n_estimators=1000, criterion="entropy", 
                                  max_features=3, min_impurity_decrease=0.0, 
                                  min_samples_split=3, min_samples_leaf=2, 
                                  oob_score=True, random_state=constants.SEED, 
                                  class_weight=None, max_depth=None)
rf_model.fit(X_train, y_train)
print("OOB Score (good): %.3f" % rf_model.oob_score_)
y_pred = rf_model.predict(X_test)
rf_model_metrics = metrics.classification_report(y_test, y_pred)
print(rf_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
twoclass_rf = ax.get_figure()
twoclass_rf.savefig(constants.FILE_DIR + "twoclass_rf.jpg")

loss_rf_bin = (confmat[0][1]+confmat[1][0])/480.
print(loss_rf_bin)
confint = [loss_rf_bin-constants.EPS,loss_rf_bin+constants.EPS]
confint = [0.,loss_rf_bin+constants.EPS]
print(confint)


## BINARY CLASSIFICATION (SUPPORT VECTOR MACHINES)
X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=0.1, random_state=123)
svm_model = svm.SVC(kernel='poly')
svm_model.fit(X_train, y_train)
y_vpred = svm_model.predict(X_val)
svm_model_vmetrics = metrics.classification_report(y_val, y_vpred)
print(svm_model_vmetrics)
confmatv = metrics.confusion_matrix(y_val, y_vpred)
print(confmatv)
y_pred = svm_model.predict(X_test)
svm_model_metrics = metrics.classification_report(y_test, y_pred)
print(svm_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
twoclass_svmpoly = ax.get_figure()
twoclass_svmpoly.savefig(constants.FILE_DIR + "twoclass_svmpoly.jpg")

loss_rbfsvm_bin = (confmat[0][1]+confmat[1][0])/480.
print(loss_rbfsvm_bin)
confint_poly = [loss_rbfsvm_bin-constants.EPS,loss_rbfsvm_bin+constants.EPS]
confint_poly = [0,loss_rbfsvm_bin+constants.EPS]
print(confint_poly)

x = ["Random Forest", "Polynomial SVM"]
y = [loss_rf_bin, loss_rbfsvm_bin]
error_graph = plt.errorbar(x, y, yerr=constants.EPS, marker='o', fmt='.', clip_on=False)
plt.ylim(bottom=0)
plt.xlabel("Algorithms")
plt.ylabel("True Risk / Error")
plt.savefig(constants.FILE_DIR + "binary_errorgraph1.jpg")

svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_vpred = svm_model.predict(X_val)
svm_model_vmetrics = metrics.classification_report(y_val, y_vpred)
print(svm_model_vmetrics)
confmatv = metrics.confusion_matrix(y_val, y_vpred)
print(confmatv)
y_pred = svm_model.predict(X_test)
svm_model_metrics = metrics.classification_report(y_test, y_pred)
print(svm_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
twoclass_svmrbf = ax.get_figure()
twoclass_svmrbf.savefig(constants.FILE_DIR + "twoclass_svmrbf.jpg")

loss_rbfsvm_bin = (confmat[0][1]+confmat[1][0])/480.
print(loss_rbfsvm_bin)
confint_rbf = [loss_rbfsvm_bin-constants.EPS,loss_rbfsvm_bin+constants.EPS]
print(confint_rbf)

x = ["Random Forest", "RBF SVM"]
y = [loss_rf_bin, loss_rbfsvm_bin]
error_graph = plt.errorbar(x, y, yerr=constants.EPS, marker='o', fmt='.', clip_on=False)
plt.ylim(bottom=0)
plt.xlabel("Algorithms")
plt.ylabel("True Risk / Error")
plt.savefig(constants.FILE_DIR + "binary_errorgraph2.jpg")


## MULTI-CLASS CLASSIFICATION [3 CLASSES] (RANDOM FOREST)
wine_data['quality_ternary'] = pd.cut(wine_data['quality'],
         bins=constants.TERNARY_BINS, labels=constants.TGROUP_NAMES)
quality_ternary = LabelEncoder()
wine_data['quality_ternary'] = quality_ternary.fit_transform(
        wine_data['quality_ternary'])
X = wine_data.drop(columns=['quality'])
y = pd.Series.to_frame(wine_data['quality_ternary'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=123)

rf_model = RandomForestClassifier(n_estimators=1000, criterion="entropy", 
                                  max_features=3, min_impurity_decrease=0.0, 
                                  min_samples_split=3, min_samples_leaf=2, 
                                  oob_score=True, random_state=constants.SEED, 
                                  class_weight=None, max_depth=None)
rf_model.fit(X_train, y_train)
print("OOB Score: %.3f" % rf_model.oob_score_)
y_pred = rf_model.predict(X_test)
rf_model_metrics = metrics.classification_report(y_test, y_pred)
print(rf_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values") 
plt.ylabel("Predicted Values")
threeclass_rf = ax.get_figure()
threeclass_rf.savefig(constants.FILE_DIR + "threeclass_rf.jpg")

loss_rf_tern = confmat[2][1]/480.
print(loss_rf_tern)
confint_trbf = [loss_rf_tern-constants.EPS,loss_rf_tern+constants.EPS]
print(confint_trbf)

X = wine_data.drop(columns=["quality"])
y = pd.Series.to_frame(wine_data['quality'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=123)
rf_model = RandomForestClassifier(n_estimators=1500, criterion="entropy", 
                                  max_features=5, min_impurity_decrease=0.0, 
                                  min_samples_split=3, min_samples_leaf=2, 
                                  oob_score=True, random_state=constants.SEED, 
                                  class_weight=None, max_depth=None)
rf_model.fit(X_train, y_train)
print("OOB Score: %.3f" % rf_model.oob_score_)
y_pred = rf_model.predict(X_test)
rf_model_metrics = metrics.classification_report(y_test, y_pred)
print(rf_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
allclass_rf = ax.get_figure()
allclass_rf.savefig(constants.FILE_DIR + "allclass_rf.jpg")

loss_all_classes_rbf = (confmat[0][2]+confmat[1][2]+confmat[1][3]+confmat[2][3]+
                        confmat[2][4]+confmat[3][2]+confmat[3][4]+confmat[4][3]+confmat[5][3]+confmat[5][4])/480.
print(loss_all_classes_rbf)
confint_allclasses_rbf = [loss_all_classes_rbf-constants.EPS,loss_all_classes_rbf+constants.EPS]
print(confint_allclasses_rbf)

## MULTI-CLASS CLASSIFICATION [3 CLASSES] (SUPPORT VECTOR MACHINES)
X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=0.1, random_state=123)
svm_model = svm.SVC(kernel='poly')
svm_model.fit(X_train, y_train)
y_vpred = svm_model.predict(X_val)
svm_model_vmetrics = metrics.classification_report(y_val, y_vpred)
print(svm_model_vmetrics)
confmatv = metrics.confusion_matrix(y_val, y_vpred)
print(confmatv)
y_pred = svm_model.predict(X_test)
svm_model_metrics = metrics.classification_report(y_test, y_pred)
print(svm_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
threeclass_svmpoly = ax.get_figure()
threeclass_svmpoly.savefig(constants.FILE_DIR + "threeclass_svmpoly.jpg")

loss_svm_t = confmat[2][1]/480.
print(loss_svm_t)
confint_tsvm = [loss_svm_t-constants.EPS,loss_svm_t+constants.EPS]
print(confint_tsvm)

x = ["Random Forest", "Polynomial SVM"]
y = [loss_rf_tern, loss_svm_t]
error_graph = plt.errorbar(x, y, yerr=constants.EPS, marker='o', fmt='.', clip_on=False)
plt.ylim(bottom=0)
plt.xlabel("Algorithms")
plt.ylabel("True Risk / Error")
plt.savefig(constants.FILE_DIR + "3classes_errorgraph1.jpg")

svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_vpred = svm_model.predict(X_val)
svm_model_vmetrics = metrics.classification_report(y_val, y_vpred)
print(svm_model_vmetrics)
confmatv = metrics.confusion_matrix(y_val, y_vpred)
print(confmatv)
y_pred = svm_model.predict(X_test)
svm_model_metrics = metrics.classification_report(y_test, y_pred)
print(svm_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
threeclass_svmrbf = ax.get_figure()
threeclass_svmrbf.savefig(constants.FILE_DIR + "threeclass_svmrbf.jpg")

loss_svm_trbf = (confmat[0][1]+confmat[1][0]+confmat[2][1])/480.
print(loss_svm_trbf)
confint_tsvmrbf = [loss_svm_trbf-constants.EPS,loss_svm_trbf+constants.EPS]
print(confint_tsvmrbf)

x = ["Random Forest", "RBF SVM"]
y = [loss_rf_tern, loss_svm_trbf]
error_graph = plt.errorbar(x, y, yerr=constants.EPS, marker='o', fmt='.', clip_on=False)
plt.ylim(bottom=0)
plt.xlabel("Algorithms")
plt.ylabel("True Risk / Error")
plt.savefig(constants.FILE_DIR + "3classes_errorgraph2.jpg")

X = wine_data.drop(columns=["quality"])
y = pd.Series.to_frame(wine_data['quality'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=123)
X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=0.1, random_state=123)
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_vpred = svm_model.predict(X_val)
svm_model_vmetrics = metrics.classification_report(y_val, y_vpred)
print(svm_model_vmetrics)
confmatv = metrics.confusion_matrix(y_val, y_vpred)
print(confmatv)
y_pred = svm_model.predict(X_test)
svm_model_metrics = metrics.classification_report(y_test, y_pred)
print(svm_model_metrics)
confmat = metrics.confusion_matrix(y_test, y_pred)
print(confmat)
ax = sns.heatmap(confmat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
allclass_svmrbf = ax.get_figure()
allclass_svmrbf.savefig(constants.FILE_DIR + "allclass_svmrbf.jpg")

loss_all_classes_svm = (confmat[0][2]+confmat[1][2]+confmat[1][3]+confmat[1][4]+confmat[2][3]+
                        confmat[2][4]+confmat[3][2]+confmat[3][4]+confmat[4][2]+confmat[4][3]+confmat[5][3]+confmat[5][4])/480.
print(loss_all_classes_svm)
confint_allclasses_svm = [loss_all_classes_svm-constants.EPS,loss_all_classes_svm+constants.EPS]
print(confint_allclasses_svm)

x = ["Random Forest", "RBF SVM"]
y = [loss_all_classes_rbf, loss_all_classes_svm]
error_graph = plt.errorbar(x, y, yerr=constants.EPS, marker='o', fmt='.', clip_on=False)
plt.xlabel("Algorithms")
plt.ylabel("True Risk / Error")
plt.savefig(constants.FILE_DIR + "allclasses_errorgraph1.jpg")