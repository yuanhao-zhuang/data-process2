# data-process2

import os

import numpy as np

import pandas as pd

import seaborn as sns

from scipy.stats import norm

from collections import Counter

import matplotlib.pyplot as plt

import sklearn

from sklearn import tree

from sklearn.svm import SVC

from sklearn.metrics import roc_curve

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import StackingClassifier

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold, cross_validate

from sklearn.metrics import recall_score, f1_score, roc_auc_score

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

import xgboost as xgb

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit








data=pd.read_csv("Desktop\论文材料\data.csv")

data





cor_matrix = data.corr().abs()

cor_matrix.style.background_gradient(sns.light_palette('red', as_cmap=True))





upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

dropped_cols = set()

for feature in upper_tri.columns:
    if any(upper_tri[feature] > 0.9): 
        dropped_cols.add(feature)
        
print("There are %d dropped columns" %len(dropped_cols))

no_correlated_data = data.drop(dropped_cols,axis=1)

no_correlated_data.head()



labels = no_correlated_data['Bankrupt?']
features = no_correlated_data.drop(['Bankrupt?'], axis = 1)

from sklearn.decomposition import PCA
n_components = 10
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(features)
features_pc = pd.DataFrame(data=principal_components, columns=['PC %d'%d for d in range(n_components)])
print("Explained variance by 10 components %.2f" %sum(pca.explained_variance_ratio_))


features_pc

new_df=pd.concat([pd.DataFrame(labels),features_pc],axis=1,sort=False)
new_df
labels = new_df['Bankrupt?']
last_features= new_df.drop(['Bankrupt?'], axis = 1)





X_raw,X_test,y_raw,y_test  = train_test_split(last_features,
                                              labels,
                                              test_size=0.2,
                                              stratify = labels,
                                              random_state = 42)

sss = StratifiedKFold(n_splits=5,random_state=None,shuffle=False)
for train_index, test_index in sss.split(X_raw,y_raw):    
    print("Train:", train_index, "Test:", test_index)
    X_train_sm, X_val_sm = X_raw.iloc[train_index], X_raw.iloc[test_index]
    y_train_sm, y_val_sm = y_raw.iloc[train_index], y_raw.iloc[test_index]
X_train_sm = X_train_sm.values
X_val_sm = X_val_sm.values
y_train_sm = y_train_sm.values
y_val_sm = y_val_sm.values
train_unique_label, train_counts_label = np.unique(y_train_sm, return_counts=True)
test_unique_label, test_counts_label = np.unique(y_val_sm, return_counts=True)
print('-' * 84)
print('Label Distributions: \n')
print(train_counts_label/ len(y_train_sm))
print(test_counts_label/ len(y_val_sm))





accuracy_lst_reg = []
precision_lst_reg = []
recall_lst_reg = []
f1_lst_reg = []
auc_lst_reg = []
log_reg_sm = LogisticRegression()
#log_reg_params = {}
log_reg_params = {"penalty": ['l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'class_weight': ['balanced',None],
                  'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)
for train, val in sss.split(X_train_sm, y_train_sm):
    pipeline_reg = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) 
    model_reg = pipeline_reg.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg = rand_log_reg.best_estimator_
    prediction_reg = best_est_reg.predict(X_train_sm[val])
    accuracy_lst_reg.append(pipeline_reg.score(X_train_sm[val], y_train_sm[val]))
    precision_lst_reg.append(precision_score(y_train_sm[val], prediction_reg))
    recall_lst_reg.append(recall_score(y_train_sm[val], prediction_reg))
    f1_lst_reg.append(f1_score(y_train_sm[val], prediction_reg))
    auc_lst_reg.append(roc_auc_score(y_train_sm[val], prediction_reg))
print('---' * 45)
print('')
print('Logistic Regression (SMOTE) results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_lst_reg)))
print("precision: {}".format(np.mean(precision_lst_reg)))
print("recall: {}".format(np.mean(recall_lst_reg)))
print("f1: {}".format(np.mean(f1_lst_reg)))
print('')
print('---' * 45)




len(y_train_sm[train])
len(X_train_sm[train])




label = ['Fin.Stable', 'Fin.Unstable']
pred_reg_sm = best_est_reg.predict(X_val_sm)
print(classification_report(y_val_sm, pred_reg_sm, target_names=label))





y_score_reg = best_est_reg.predict(X_val_sm)
average_precision = average_precision_score(y_val_sm, y_score_reg)
fig = plt.figure(figsize=(12,6))
precision, recall, _ = precision_recall_curve(y_val_sm, y_score_reg)
plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
          average_precision), fontsize=15)
plt.show()





















