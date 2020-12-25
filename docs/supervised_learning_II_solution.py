'''
supervised learning II
@Jimmy Azar
'''

#Exercise 1 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path_to_file = './data/boomerang2D.txt'
data = pd.read_csv(path_to_file, sep=' ')

X = data.drop('label',axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

plt.scatter(X_train['f1'], X_train['f2'], c=y_train)
plt.show() 

#sklearn 
from sklearn.neural_network import MLPClassifier #default activation is 'relu' for the hidden layer.
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30, 30), activation='logistic') #hidden_layer_sizes=(15,) i.e. 15 units in 1 layer only
#solver 'adam' works poorly on small datasets and well on large ones. 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)         
model.score(X_test, y_test) #accuracy
n_error = (y_pred!=y_test).sum() 

#decision boundary 
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,step), np.arange(y_min,y_max,step))

labels_pred = model.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4) 
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train.replace({1:'black',2:'red'})) #plot this last
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred_prob[:,0], pos_label=1) #inputs should be array, not array-like
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC={roc_auc.round(2)}')
plt.plot([0, 1], [0, 1], c='gray', lw=1, linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.show()

#Exercise 2
path_to_file = './data/boomerang2Dlarge.txt'
data = pd.read_csv(path_to_file, sep=' ')

X = data.drop(['label'],axis=1)
y = data['label']

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda  

n_train = [10, 20, 50, 100]
error_test_mean, error_test_std = [], []
for n in n_train:
	error_test = []
	for i in range(10):
		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, stratify=y)
		model = qda().fit(X_train, y_train) #can change to lda()  
		error_test.append(1 - model.score(X_test, y_test))
		#y_pred = model.predict(X_test) #alternatively
		#error_test.append((y_pred!=y_test).mean())
	
	error_test_mean.append(np.mean(error_test))	
	error_test_std.append(np.std(error_test))

plt.errorbar(n_train, error_test_mean, error_test_std, lw=2)
plt.xlabel('training set size')
plt.ylabel('classification error rate')
plt.show()

#alternative: using sklearn (skip, method learning_curve() contains bugs)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=qda(), X=X, y=y, train_sizes=n_train, cv=3, shuffle=True) #CV used over X, y
#n_train should have large enough sizes and cv should be small enough too, otherwise can end up with 1 class in a fold
#shuffle is necessary because implementation contains a bug (see below url).

'''
https://github.com/scikit-learn/scikit-learn/issues/9913

Exactly. If the input is sorted, it fails for SVC. I dived in the code a bit but it's hard to follow how the cv splits for the curve are actually done. I think at least a hint in the documentation is need because for RandomForest it's obviously even worse because it doesn't fail but the curve is generated from "weird" CV folds.

stratifiedkfold is used for the cv part, but not for the learning curve part. It used to randomize, but there was an issue about using it on time series, and I think we decided not to shuffle by default. We should probably add a sentence to the docs that you should think about whether you want to shuffle or not. We don't shuffle in CV by default, so I think not shuffling is consistent, but maybe we should explain the subsetting strategy better in the docs.

When using some random state such as 0,100, 80, the learning_curve worked perfectly. However, in case of some random state such as 10, 20 (of course, this depends on my specific dataset), I got the error message. Setting "shuffle" parameter to True didn't help either. The above observations were done using cv=ShuffleSplit. When using cv=StratifiedKFold, it seemed that learning_curve didn't work at all.
'''

train_mean = np.mean(1-train_scores, axis=1)
train_std = np.std(1-train_scores, axis=1)
test_mean = np.mean(1-test_scores, axis=1)
test_std = np.std(1-test_scores, axis=1)

plt.errorbar(train_sizes, test_mean, test_std, lw=2)
plt.xlabel('training set size')
plt.ylabel('classification error rate')
plt.grid()
plt.show()

#Exercise 3
path_to_file = './data/boomerang2Dlarge.txt'
data = pd.read_csv(path_to_file, sep=' ')

X = data.drop(['label'],axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, stratify=y)

y_train.value_counts() 

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#model = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train) #model2 = LinearSVC().fit(X_train, y_train)  #alternatively: linear SVC
model = SVC(kernel='rbf', C=1, gamma=1, probability=True).fit(X_train, y_train)

y_pred = model.predict(X_test) 
model.score(X_test, y_test)
confusion_matrix(y_true=y_test, y_pred=y_pred)

#decision boundary 
step = 0.1
x_min, x_max = X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1
y_min, y_max = X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,step), np.arange(y_min,y_max,step))

labels_pred = model.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = labels_pred.reshape(xx.shape)

plt.scatter(xx.ravel(), yy.ravel(), c=labels_pred, marker='+', alpha=0.4) 
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train.replace({1:'black',2:'red'})) 
plt.show()

#Exercise 4
model_qda = qda().fit(X_train, y_train)
y_prob_qda = model_qda.predict_proba(X_test)
y_pred_qda = model_qda.predict(X_test)
confusion_matrix(y_true=y_test, y_pred=y_pred)

model_lda = lda().fit(X_train, y_train)
y_prob_lda = model_lda.predict_proba(X_test)
y_pred_lda = model_lda.predict(X_test)
confusion_matrix(y_true=y_test, y_pred=y_pred)

y_prob_product = y_prob_qda * y_prob_lda
y_pred_product = y_prob_product.argmax(axis=1) + 1

(y_pred_product != y_pred_qda).sum() # = 2 
(y_pred_product != y_pred_lda).sum() # = 2 

#only the sum rule is available in sklearn through the method VotingClassifier() with 'soft' option 
from sklearn.ensemble import VotingClassifier
model_sum = VotingClassifier(estimators=[('qda', model_qda), ('lda', model_lda)], voting='soft')
model_sum = model_sum.fit(X_train, y_train)
y_pred_sum = model_sum.predict(X_test)

#Exercise 5
def synth_data(n_samples, sigma):
	X = np.random.uniform(low=0,high=2,size=n_samples)
	X = np.sort(X)
	y = np.sin(4*X) + np.random.normal(loc=0,scale=sigma,size=n_samples) 
	return X, y
	
X_train, y_train = synth_data(n_samples=50, sigma=0.1)

plt.scatter(X_train, y_train)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

from sklearn.svm import SVR
model = SVR(kernel='rbf', C=1, epsilon=0.1, gamma='scale').fit(X_train.reshape(-1,1), y_train) #1D array not allowed: reshape()
y_pred = model.predict(X_train.reshape(-1,1))

plt.scatter(X_train, y_train, c='black', label='original')
plt.scatter(X_train, y_pred, c='red', marker='x', lw=2, label='SVR predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show() 

rmse = np.sqrt(((y_train - y_pred)**2).mean())

#Exercise 6
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ['rbf'], 
			  'gamma': np.linspace(1,10,20),
			  'epsilon': np.linspace(0,1,10),
			  'C': 2**np.arange(0,10)}
  
model = GridSearchCV(SVR(), parameters, scoring='neg_mean_squared_error', cv=5) #neg_mean_squared_error is used because the method selects model with max criterion by default
model.fit(X_train.reshape(-1,1), y_train)
model.best_params_
#model.best_estimator_
#model.best_score_
y_pred = model.predict(X_train.reshape(-1,1)) 

rmse = np.sqrt(((y_train - y_pred)**2).mean()) #less than before (0.07<0.09)

plt.scatter(X_train, y_train, c='black', label='original')
plt.scatter(X_train, y_pred, c='red', marker='x', lw=2, label='SVR predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show() 

#Exercise 7
X_test = np.arange(-1,3,0.05)
y_pred = model.predict(X_test.reshape(-1,1))

plt.scatter(X_train, y_train, c='black', label='original')
plt.scatter(X_test, y_pred, c='red', marker='x', lw=2, label='SVR predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show() 
