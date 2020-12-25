'''
feature selection & extraction 
@Jimmy Azar
'''

#Exercise 1
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

iris = datasets.load_iris()
meas = iris.data        #ndarray
y = iris.target         #array
X = np.random.multivariate_normal([0]*10,np.diag([1]*10),150)
X = pd.DataFrame(X,columns=[f'V{i}' for i in range(10)])  #or ['V'+str(i) for i in range(10)]

X[['V0','V2','V4','V6']] = meas  #or X.iloc[:,[0,2,4,6]]

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=4, step=1)
rfe.fit(X, y)
rfe.support_.nonzero()     #or np.where(ref.support_)

#Exercise 2 
path_to_file = './data/sonar.txt'
a = pd.read_csv(path_to_file,sep=' ')

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA() #n_components not specified => keep all
pca.fit(a)
rv = pca.explained_variance_.cumsum()/pca.explained_variance_.sum()  #in R's princomp() division is by n whereas here by (n-1): so, to get these do: b$sdev^2*208/207 in R.
pca.explained_variance_ratio_  #same as pca.explained_variance_/sum(pca.explained_variance_)
V = pca.components_            #components are on rows
pca.fit_transform(a)           #same as R's ...$scores (i.e. new coordinates in transformed feature space)

plt.plot(np.arange(1,60+1),rv,linewidth=2,color='blue')
plt.xlabel('number of features')
plt.ylabel('ratio of retained variance to total variance')
plt.show()
    
#Exercise 3 
from sklearn.preprocessing import StandardScaler
a_scaled = StandardScaler().fit_transform(a)  #uses numpy 
#a_scaled = (a-a.mean())/a.std(ddof=0)        #pandas divides by n whereas numpy's np.std() divides by (n-1)

pca = PCA().fit(a_scaled)
rv_scaled = pca.explained_variance_.cumsum()/pca.explained_variance_.sum()

plt.plot(np.arange(1,60+1),rv,linewidth=2,color='red')
plt.plot(np.arange(1,60+1),rv_scaled,linewidth=2,color='black')
plt.xlabel('number of features')
plt.ylabel('ratio of retained variance to total variance')
plt.legend(['original','normalized'])
plt.show()

#Exercise 4 
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data           #ndarray
y = iris.target         #array

import inspect 
print(inspect.getsource(sklearn.discriminant_analysis.LinearDiscriminantAnalysis.transform)) #to view source code

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  #sklearn.lda was removed in later versions of scikit-learn
target_names = iris.target_names

lda = LDA()  
X_new = lda.fit(X, y).transform(X)
lda.fit(X, y).scalings_   #these are the coefficients

#plt.scatter(X_new[:,0],X_new[:,1],color=pd.DataFrame(y).replace({0:'black',1:'red',2:'green'})[0])  #convert numpy to DataFrame to use replace()
plt.scatter(X_new[:,0],X_new[:,1],c=y) 
plt.ylabel('Feature 2')
plt.show()

#if legend required
'''
target_names = iris.target_names
for c, i, target_name in zip(['r','g','b'], [0, 1, 2], target_names):   
    plt.scatter(X_new[y == i, 0], X_new[y == i, 1], color=c, label=target_name)
plt.legend()
plt.show()
'''

#Exercise 5
path_to_file = './data/circles.txt'
a = pd.read_csv(path_to_file,sep=' ')

#3d plot
ax = plt.axes(projection='3d')
ax.scatter(a['f1'],a['f2'],a['f3'])
plt.show()

#pca
X_pca = PCA(n_components=2).fit_transform(a) 

plt.scatter(X_pca[:,0],X_pca[:,1],color='blue',marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

#Exercise 6

#import os
#os.getcwd()

from sklearn.manifold import MDS
#from sklearn.metrics import euclidean_distances     #not needed since mainfold.MDS() can take 'euclidean' option  
#similarities = euclidean_distances(a[['f1','f2']])  #like R's pdist()

mds = MDS(n_components=2, max_iter=3000, dissimilarity="euclidean")
pos = mds.fit_transform(a)   

plt.figure()
plt.scatter(pos[:,0],pos[:,1],marker='+') 
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

#Exercise 7
path_to_file = './data/lines.txt'
a = pd.read_csv(path_to_file,sep=' ')

#3d plot
ax = plt.axes(projection='3d')
ax.scatter(a['f1'],a['f2'],a['f3'])
plt.show()

#pca
X_pca = PCA(n_components=2).fit_transform(a) 

plt.scatter(X_pca[:,0],X_pca[:,1],color='blue',marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

#Exercise 8
mds = MDS(n_components=2, max_iter=3000, dissimilarity="euclidean")
pos = mds.fit_transform(a)   

plt.figure()
plt.scatter(pos[:,0],pos[:,1],marker='+') 
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axis('equal')
plt.show()

#Exercise 9
def Rastrigin(x):
	err = 20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
	return err

x1 = np.linspace(-3,3,100)
x2 = np.linspace(-3,3,100)

#method 1: reshape 1D array
e = np.array([Rastrigin([i,j]) for i in x1 for j in x2]).reshape(len(x1),len(x2))
e = pd.DataFrame(e)

#method 2: tuple to DataFrame then pivoting 
#q = [(i,j,Rastrigin([i,j])) for i in x1 for j in x2] 
#e = pd.DataFrame(q).pivot(index=0, columns=1, values=2)

#surface plot
import matplotlib.pyplot as plt

ax = plt.axes(projection="3d")
x, y = np.meshgrid(x1, x2)
ax.plot_surface(x,y,e)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

#heatmap/contours
ticks_loc = np.arange(0,len(e.columns),5)
plt.xticks(ticks=ticks_loc, labels=x1.round(2)[ticks_loc], rotation=90)
plt.yticks(ticks=ticks_loc, labels=x2.round(2)[ticks_loc])
plt.imshow(e,cmap='hot') 
plt.colorbar()
plt.show()

#Exercise 10/11
#pip3 install geneticalgorithm 
from geneticalgorithm import geneticalgorithm as ga  #https://pypi.org/project/geneticalgorithm/

varbound = np.array([[-3,3]]*2)
algorithm_param = {'max_num_iteration': 1000,
                   'population_size':20,
                   'mutation_probability':0.2,
                   'elit_ratio':2/100,
                   'crossover_probability':0.8,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}
model = ga(function=Rastrigin, dimension=2, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model.run()

#prob. of crossover, prob. of mutation, 100 runs without improvement in fitness value 
#it does minimization of the function by default

#Exercise 12
# When there is a prevalence of local minima in the objective function
# When it is difficult or impossible to implement gradient descent such as: 
# When the error function is discontinuous, non-differentiable, stochastic, or highly nonlinear.
