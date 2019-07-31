#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:53:55 2019

@author: David_Tsai
"""
#import libraries
import numpy as np
from numpy import loadtxt
from numpy import sort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.multiclass import unique_labels
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from pcm import plot_confusion_matrix

# ---------------------data preprocessing----------------------- 
df_orginal = pd.read_csv('107MDS_telecom.csv')
df = df_orginal.iloc[:,0:3]

# rename from CHN to Eng
df.rename(columns={'區域':'Area'}, inplace=True)
df['Area'] = df['Area'].replace({'北區': 'N', '中區': 'C', '南區': 'S'})



# EDA
ax = sns.countplot(x="q01v01", data=df)
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 

ax = sns.countplot(x="q01", data=df)
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 

ax = sns.countplot(x="Area", data=df)
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 
    
pivot_df = df.pivot_table(df, index='Area', columns='q01',aggfunc=len)
pivot_df.columns = ['1', '2', '3', '5', '6', '7', '99']
pivot_df = pivot_df.reindex(['N', 'C', 'S'])
pivot_df.rename(index = {"N": "North", 
                     "C":"Central",
                     "S":"South"}, 
                                 inplace = True) 
pivot_df.plot.bar(stacked=True, figsize=(10,7))


g = sns.catplot(x="q01", hue="q01v01", col="Area",
             data=df, kind="count")

g = sns.catplot(x="Area", hue="q01v01", col="q01",
             data=df, kind="count")


df = df[df.q01v01 != 0]
df = pd.get_dummies(df, columns = ['Area'])

X = df.drop(columns = ['q01v01'])
y = df['q01v01']


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

#X_train = pd.DataFrame(data= X_train)
#y_train = pd.DataFrame(data= y_train)
#
#X_train.columns = ['q01', 'Area_M', 'Area_N', 'Area_S']
#y_train.columns = ['q01v01']

# -------------------------- Xgboost------------------------------------
xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# ROC
predicted_probas = xgb.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# plot feature importance
from xgboost import plot_importance
plot_importance(xgb)
plt.show()

#prediction
y_pred = xgb.predict(X_test)
predictions = [round(value) for value in y_pred]

cks_xgb = cohen_kappa_score(y_test, y_pred)

from sklearn.metrics import f1_score
f1_xgb = f1_score(y_test, y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm = cm, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "XGboost CM")

# skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

#classification report
from sklearn.metrics import classification_report
cr_xgb = classification_report(y_test, y_pred)
print(cr_xgb)


visualizer = ClassificationReport(xgb, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()           


#-------------------------- Random Forest------------------------------
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train)

# ROC
predicted_probas = rf.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Predicting the Test set results
y_pred_rf = rf.predict(X_test)
predictions = [round(value) for value in y_pred_rf]

cks_rf = cohen_kappa_score(y_test, y_pred_rf)

from sklearn.metrics import f1_score
f1_rf = f1_score(y_test, y_pred_rf)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

plot_confusion_matrix(cm = cm_rf, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "Random Forest CM")

# skplt.metrics.plot_confusion_matrix(y_test, y_pred_rf, normalize=True)

#classification report
from sklearn.metrics import classification_report
cr_rf = classification_report(y_test, y_pred_rf)
print(cr_rf)

visualizer = ClassificationReport(rf, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()         

#--------------------Balanced Random Forest------------------------------
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(class_weight= "balanced", random_state = 0)
brf.fit(X_train, y_train)

# ROC
predicted_probas = brf.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Predicting the Test set results
y_pred_brf = brf.predict(X_test)
predictions = [round(value) for value in y_pred_brf]

from sklearn.metrics import f1_score
f1_brf = f1_score(y_test, y_pred_brf)

cks_brf = cohen_kappa_score(y_test, y_pred_brf)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_brf = confusion_matrix(y_test, y_pred_brf)

plot_confusion_matrix(cm = cm_brf, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "Balanced Random Forest CM")

#classification report
from sklearn.metrics import classification_report
cr_brf = classification_report(y_test, y_pred_brf)
print(cr_brf)

visualizer = ClassificationReport(brf, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()         


#------------------------ Logistic Regression--------------------------
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(class_weight = 'balanced', random_state = 0)
classifier.fit(X_train, y_train)

# ROC
predicted_probas = classifier.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Predicting the Test set results
y_pred_lr = classifier.predict(X_test)
predictions = [round(value) for value in y_pred_lr]
cks_lr = cohen_kappa_score(y_test, y_pred_lr)

from sklearn.metrics import f1_score
f1_lr = f1_score(y_test, y_pred_lr)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

plot_confusion_matrix(cm = cm_lr, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "Logistic Regression CM")

#classification report
from sklearn.metrics import classification_report
cr_lr = classification_report(y_test, y_pred_lr)
print(cr_lr)

visualizer = ClassificationReport(classifier, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()         

# ----------------------------Naive Bayes-------------------------------------
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train, y_train)

# ROC
predicted_probas = NB.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Predicting the Test set results
y_pred_nb= NB.predict(X_test)
predictions_nb = [round(value) for value in y_pred_nb]

from sklearn.metrics import f1_score
f1_nb = f1_score(y_test, y_pred_nb)

cks_nb = cohen_kappa_score(y_test, y_pred_nb)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)

plot_confusion_matrix(cm = cm_nb, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "Naive Bayes CM")

#classification report
from sklearn.metrics import classification_report
cr_nb = classification_report(y_test, y_pred_nb)
print(cr_nb)

visualizer = ClassificationReport(NB, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()         


# -----------------------------KNN-------------------------------------------
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
KNN.fit(X_train, y_train)

# ROC
predicted_probas = KNN.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Predicting the Test set results
y_pred_knn = KNN.predict(X_test)
predictions_knn = [round(value) for value in y_pred_knn]

from sklearn.metrics import f1_score
f1_knn = f1_score(y_test, y_pred_knn)

cks_knn = cohen_kappa_score(y_test, y_pred_knn)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

plot_confusion_matrix(cm = cm_knn, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "KNN CM")

#classification report
from sklearn.metrics import classification_report
cr_knn = classification_report(y_test, y_pred_knn)
print(cr_knn)

visualizer = ClassificationReport(KNN, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()         

#------------------------------SVM--------------------------------------------
# Fitting SVM to the Training set
from sklearn.svm import SVC
SVC= SVC(kernel = 'linear', random_state = 0)
SVC.fit(X_train, y_train)

# ROC
predicted_probas = SVC.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Predicting the Test set results
y_pred_svc = SVC.predict(X_test)
predictions_svc = [round(value) for value in y_pred_svc]
cks_svm = cohen_kappa_score(y_test, y_pred_svc)

from sklearn.metrics import f1_score
f1_svc = f1_score(y_test, y_pred_svc)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm = cm_svc, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "SVM CM")

#classification report
from sklearn.metrics import classification_report
cr_svm = classification_report(y_test, y_pred_svc)
print(cr_svm)

visualizer = ClassificationReport(SVC, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()     

#------------------------------Kernel SVM--------------------------------------------
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', random_state = 0)
ksvm.fit(X_train, y_train)

# ROC
predicted_probas = ksvm.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Predicting the Test set results
y_pred_ksvm = classifier.predict(X_test)
predictions_ksvm = [round(value) for value in y_pred_ksvm]
cks_ksvm = cohen_kappa_score(y_test, y_pred_ksvm)

from sklearn.metrics import f1_score
f1_ksvm = f1_score(y_test, y_pred_ksvm)
                  
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ksvm = confusion_matrix(y_test, y_pred_ksvm)

plot_confusion_matrix(cm = cm_ksvm, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "Kernel SVM CM")


#classification report
from sklearn.metrics import classification_report
cr_ksvm = classification_report(y_test, y_pred_ksvm)
print(cr_ksvm)

visualizer = ClassificationReport(ksvm, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()   


#------------------------------Adaboost--------------------------------------------
# Fitting Adaboost to the Training set
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc.fit(X_train, y_train)

# ROC
predicted_probas = abc.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

#Precision Recall Curve
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# Predicting the Test set results
y_pred_abc = abc.predict(X_test)
predictions_ksvm = [round(value) for value in y_pred_abc]
cks_abc = cohen_kappa_score(y_test, y_pred_abc)

from sklearn.metrics import f1_score
f1_abc = f1_score(y_test, y_pred_abc)
                  
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_abc = confusion_matrix(y_test, y_pred_abc)

plot_confusion_matrix(cm = cm_abc, normalize= False,
                      target_names= ['Buy', 'No Buy'], title= "Adaboost CM")


#classification report
from sklearn.metrics import classification_report
cr_abc = classification_report(y_test, y_pred_abc)
print(cr_abc)

visualizer = ClassificationReport(abc, classes=['Buy', 'No Buy'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  
g = visualizer.poof()   


cksdata = {'Types of Algorithm' :['Adaboost', 'Logistic Regression', 'Naive Bayes', 'Random Forest','Xgboost'],
        'Cohen Kappa Score':[cks_abc, cks_lr, cks_nb, cks_rf,cks_xgb]}
coka = pd.DataFrame(data = cksdata)

cksx = sns.barplot(x= 'Types of Algorithm', y= 'Cohen Kappa Score', data=coka)


f1data = {'Types of Algorithm' :['Adaboost', 'Logistic Regression', 'Naive Bayes', 'Random Forest','Xgboost'],
        'F1 Score':[f1_abc, f1_lr, f1_nb, f1_rf, f1_xgb]}
f1s = pd.DataFrame(data = f1data)
f1x = sns.barplot(x= 'Types of Algorithm', y= 'F1 Score', data= f1s)



