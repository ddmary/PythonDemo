
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:48:17 2018

@author: dongdongmary
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
import seaborn as sns



#loading dataset

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

#EDA

#explore the basic statistics of wine dataset
explore=df_wine.describe().T
print (explore)

#draw a boxplot of the features
sc = StandardScaler()
data_std = pd.DataFrame(sc.fit_transform(df_wine))
box = data_std.iloc[:,1:].values
plt.boxplot(box,notch = False, sym = 'rs',vert = True)
plt.title('box plot')
plt.xlabel('Features')
plt.ylabel('Features value')
plt.show()


#draw a correlation coefficient matrix with a heatmap
cm = df_wine.corr()
plt.subplots(figsize=(9, 9))
plt.title('a correlation coefficient matrix with a heatmap')
hm = sns.heatmap(cm,cbar = True,annot = True,square = True,
                 fmt = '.2f',annot_kws = {'size':10},cmap="YlGnBu")
plt.show()


# Splitting the data into 80% training and 20% test subsets.

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                     stratify=y,
                     random_state=42)
# Standardizing the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)



#Logistic regression

# Instantiate the GridSearchCV object and run the search
print('Baseline')
print('Logistic regression:')
lr = LogisticRegression()
parameters = [{'C':np.arange(0.01,2.0,0.01).tolist(), 'multi_class':['ovr']}]
clf=GridSearchCV(lr,parameters,scoring='accuracy',cv=5)

#find best model params
clf.fit(X_train_std,y_train)
print("Best CV params", clf.best_estimator_)
print()

#fit the model using best params
best_lr = clf.best_estimator_
best_lr.fit(X_train_std,y_train)

print("LR training accuracy:", best_lr.score(X_train_std, y_train))
print("LR test accuracy    :", best_lr.score(X_test_std, y_test))


#SVM MODEL
# Instantiate an RBF SVM
print()
print('SVM Model:')
#svm = SVC()

# Instantiate the GridSearchCV object and run the search
#parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
svm = SVC(C=1, kernel='linear', random_state=42)
svm.fit(X_train_std, y_train)
#searcher = GridSearchCV(svm, parameters)
#searcher.fit(X, y)


# Report the best parameters
#print("Best CV params", searcher.best_estimator_)

#fit the model using best params
#best_svm = searcher.best_estimator_
#best_svm.fit(X_train_std,y_train)


print("SVM training accuracy:", svm.score(X_train_std, y_train))
print("SVM test accuracy    :", svm.score(X_test_std, y_test))


# Principal component analysis(PCA)


print()
print('Principal component analysis(PCA):')


#PCA transform
pca = c
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

#PCA_Logistic_Regression_model
print('PCA--Logistic Regression Model :')
parameters = [{'C':np.arange(0.01,2.0,0.01).tolist(), 'multi_class':['ovr']}]
clf=GridSearchCV(lr,parameters,scoring='accuracy',cv=5)

#find best model params
clf.fit(X_train_pca,y_train)
print("Best CV params", clf.best_estimator_)
print()

#fit the model using best params
best_lr_pca = clf.best_estimator_
best_lr_pca.fit(X_train_pca,y_train)

print("PCA-LR training accuracy:", best_lr_pca.score(X_train_pca, y_train))
print("PCA-LR test accuracy    :", best_lr_pca.score(X_test_pca, y_test))

#PCA_SVM_MODEL
print('PCA--SVM Model :')
svm = SVC(C=1, kernel='linear', random_state=42)
svm.fit(X_train_pca, y_train)

print("PCA-SVM training accuracy:", svm.score(X_train_pca, y_train))
print("PCA-SVM test accuracy    :", svm.score(X_test_pca, y_test))




# linear discriminant analysis(LDA)

print()
print('linear discriminant analysis(LDA)')

#LDA transform
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)


#lda_linear_regression medel
lr = lr.fit(X_train_lda, y_train)

print('LDA--Logistic Regression Model :')
parameters = [{'C':np.arange(0.01,2.0,0.01).tolist(), 'multi_class':['ovr']}]
clf=GridSearchCV(lr,parameters,scoring='accuracy',cv=10)

#find best model params
clf.fit(X_train_lda,y_train)
print("Best CV params", clf.best_estimator_)
print()

#fit the model using best params
best_lr_lda = clf.best_estimator_
best_lr_lda.fit(X_train_lda,y_train)

print("LDA-LR training accuracy:", best_lr_lda.score(X_train_lda, y_train))
print("LDA-LR test accuracy    :", best_lr_lda.score(X_test_lda, y_test))

#LDA_SVM_MODEL
print('LDA--SVM Model :')
svm = SVC(C=1, kernel='linear', random_state=42)
svm.fit(X_train_lda, y_train)

print("LDA-SVM training accuracy:", svm.score(X_train_lda, y_train))
print("LDA-SVM test accuracy    :", svm.score(X_test_lda, y_test))


#KPCA

print()
print('kPCA model')
kpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 0.01)
X_train_kpca = kpca.fit_transform(X_train_std, y_train)
X_test_kpca = kpca.transform(X_test_std)

lr = lr.fit(X_train_kpca, y_train)

#kPCA_Logistic_Regression_model
print('kPCA--Logistic Regression Model :')
parameters = [{'C':np.arange(0.01,2.0,0.01).tolist(), 'multi_class':['ovr']}]
clf=GridSearchCV(lr,parameters,scoring='accuracy',cv=10)

#find best model params
clf.fit(X_train_kpca,y_train)
print("Best CV params", clf.best_estimator_)
print()

#fit the model using best params
best_lr_kpca = clf.best_estimator_
best_lr_kpca.fit(X_train_kpca,y_train)

print("kPCA-LR training accuracy:", best_lr_kpca.score(X_train_kpca, y_train))
print("kPCA-LR test accuracy    :", best_lr_kpca.score(X_test_kpca, y_test))


#kPCA_SVM_MODEL
print('kPCA--SVM Model :')
svm = SVC(C=1, kernel='rbf', random_state=42)
svm.fit(X_train_kpca, y_train)

print("kPCA-SVM training accuracy:", svm.score(X_train_kpca, y_train))
print("kPCA-SVM test accuracy    :", svm.score(X_test_kpca, y_test))


