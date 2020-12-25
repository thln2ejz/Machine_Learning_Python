'''
fundamental concepts 
@Jimmy Azar
'''

#Exercise 1
#clear all vars (not beginning with _)
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
# %reset -f in Jupyter

import os 
os.system('clear') #clear console

import numpy as np
import matplotlib.pyplot as plt 

x = np.random.normal(loc=0,scale=1,size=1000) 
nb = 10 
plt.hist(x,bins=nb,density=False)
plt.xlabel('x')
plt.ylabel('Frequency')
plt.show()

#Exercise 2
#Rank statistics are more robust to outliers.

#Exercise 3
y = np.random.normal(loc=0,scale=1,size=10000)
y.mean()
y.var()

fig, axs = plt.subplots(1,2)
axs[0].set_xlabel('y')
axs[0].set_ylabel('Frequency')
axs[0].hist(y,density=False)

axs[1].set_xlabel('y')
axs[1].boxplot(y,patch_artist=True,vert=True)
plt.show()
 
#Exercise 4
b = np.exp(-0.5*y)
b.mean()
b.var()

fig, axs = plt.subplots(1,2)
axs[0].set_xlabel('b')
axs[0].set_ylabel('Frequency')
axs[0].hist(b,density=False)

axs[1].set_xlabel('b')
axs[1].boxplot(b,patch_artist=False,vert=True)
plt.show()

#Exercise 5
import pandas as pd
path_to_file = './data/boomerang3D.txt'
b = pd.read_csv(path_to_file, sep=' ')  #or b = pd.read_table(path_to_file, sep=' ')

ax = plt.axes(projection='3d')
ax.scatter(b['feature1'],b['feature2'],b['feature3'],c=b['label'])
#ax.scatter(b['feature1'],b['feature2'],b['feature3'],c=np.repeat(['blue','red'],[500,500]))   #choosing colors
plt.show()

#To add legend, make a scatter plot for each class 
b1 = b.loc[b['label']==1]
b2 = b.loc[b['label']==2]
ax = plt.axes(projection='3d')
scatter1 = ax.scatter(b1['feature1'],b1['feature2'],b1['feature3'],c='blue')
scatter2 = ax.scatter(b2['feature1'],b2['feature2'],b2['feature3'],color='red')
ax.legend([scatter1, scatter2], ['label=1', 'label=2'])
plt.show()

plt.scatter(b['feature1'],b['feature2'],c=b['label'])
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.show()

#Exercise 6
path_to_file = './data/dataset8D.txt'
c = pd.read_csv(path_to_file, sep=' ')  
c.info() 
from pandas.plotting import scatter_matrix
scatter_matrix(c.iloc[:,:-1],diagonal='kde')   
plt.show()

covmat = np.cov(c.iloc[:,:-1].T).round(2)
#np.cov(c.iloc[:,:-1],rowvar=False).round(2)   #alternatively
corrmat = c.drop(['label'],axis=1).corr().round(2) #using pandas corr()
#corrmat = np.corrcoef(c.drop(['label'],axis=1),rowvar=False).round(2) #alternatively using numpy


plt.xticks(ticks=np.arange(len(c.columns)-1),labels=c.columns[:-1],rotation=90)
plt.yticks(ticks=np.arange(len(c.columns)-1),labels=c.columns[:-1])
h = plt.imshow(corrmat,cmap='hot')
plt.colorbar(h)
plt.show()

#Exercise 7
x = np.random.uniform(low=0,high=1,size=100)
x.mean() #theoretical: (b-a)/2 = 1/2 = 0.5
x.var()  #theoretical: (b-a)^2/12 = 1/12 = 0.08333

m = []
for i in enumerate(x): 
	y = np.random.uniform(low=0,high=1,size=100)
	m.append(y.mean())

m = np.array(m) 
m.mean() 
m.var() #theoretical: 0.08333/(n=100)  
print(f'mean of m: \t {np.mean(m)}') 
print(f'variance of m: \t {np.var(m)}') 
plt.hist(m,density=False)
plt.show()

#Exercise 8
n = [10,100,250,500,1000]
mean_m, mean_v, sd_m, sd_v = ([] for i in range(4));
for i in n:
	m,v = [],[]
	for j in range(1000):
		print(j)
		a = np.random.normal(loc=0,scale=1,size=i)
		m.append(a.mean())
		v.append(a.var())
	mean_m.append(np.mean(m))
	sd_m.append(np.std(m))
	mean_v.append(np.mean(v))
	sd_v.append(np.std(v))

plt.errorbar(n, mean_m, sd_m)
plt.xlabel('n')
plt.ylabel('mean_m')
plt.show()         
         
plt.errorbar(n, mean_v, sd_v)
plt.xlabel('n')
plt.ylabel('mean_v')
plt.show()         

#Exercise 9
mean = [0, 0]
cov = [[1, 0], [0, 1]]
z = np.random.multivariate_normal(mean, cov, 1000) 
C = np.cov(z,rowvar=False)
w, v = np.linalg.eig(C)

#Exercise 10 
z = np.random.multivariate_normal([0,0],[[3,0],[0,1]],1000)
C = np.cov(z.T)
w, v = np.linalg.eig(C) 

plt.scatter(z[:,0],z[:,1]) 
plt.plot([0,v[0,0]],[0,v[1,0]],color='red',linewidth=2)
plt.plot([0,v[0,1]],[0,v[1,1]],color='red',linewidth=2)
plt.axis('equal')
plt.show()

plt.scatter(z[:,0],z[:,1]) 
plt.plot(np.sqrt(w[0])*np.array([0,v[0,0]]),np.sqrt(w[0])*np.array([0,v[1,0]]),color='red',linewidth=2)
plt.plot(np.sqrt(w[1])*np.array([0,v[0,1]]),np.sqrt(w[1])*np.array([0,v[1,1]]),color='red',linewidth=2)
plt.axis('equal')
plt.show()

#Exercise 11
z = np.random.multivariate_normal([0,0],[[3,-2],[-2,2]],1000)
C = np.cov(z.T)
w, v = np.linalg.eig(C) 

plt.scatter(z[:,0],z[:,1]) 
plt.plot(np.sqrt(w[0])*np.array([0,v[0,0]]),np.sqrt(w[0])*np.array([0,v[1,0]]),color='red',linewidth=2)
plt.plot(np.sqrt(w[1])*np.array([0,v[0,1]]),np.sqrt(w[1])*np.array([0,v[1,1]]),color='red',linewidth=2)
plt.axis('equal')
plt.show()

#Exercise 12
D = np.diag(w)
w = z.dot(v).dot(np.linalg.inv(np.sqrt(D)))

plt.scatter(z[:,0],z[:,1]) 
plt.scatter(w[:,0],w[:,1],color='red')
plt.axis('equal')
plt.legend(['original','sphered'])
plt.show()

#Exercise 13
path_to_file = './data/dataset1D.txt'
data = pd.read_csv(path_to_file, sep=' ') 

a = data['a']
a.plot.kde(bw_method=0.1) 
plt.scatter(a,np.zeros(len(a)),marker='+',color='red')
plt.xlabel('a')
plt.ylabel('density')
plt.show()

from sklearn.neighbors import KernelDensity
model = KernelDensity(kernel='gaussian', bandwidth=1)
x = np.linspace(a.min()-3,a.max()+3,1000).reshape(-1,1)
model.fit(np.array(a).reshape(len(a),1))  #input needs to be 2D array 
log_dens = model.score_samples(x)         #input needs to be 2D array
y = np.exp(log_dens)

plt.plot(x,y,linewidth=2)
plt.scatter(a,np.zeros(len(a)),marker='+',color='red')
plt.xlabel('a')
plt.ylabel('density')
plt.show()

#Exercise 14
h = [0.01,0.05,0.1,0.25,0.5,1,1.5,2,3,4,5]
ll = []
for i in h:
	model = KernelDensity(bandwidth=i)
	model.fit(np.array(data['trn']).reshape(-1,1))  
	log_dens = model.score_samples(np.array(data['tst']).reshape(-1,1))
	y = np.exp(log_dens)
	ll.append(np.log(y).sum())
	
plt.plot(h,ll,color='blue',linewidth=2)  #or plt.plot(h,ll,'-o',color='blue',linewidth=2)
plt.scatter(h,ll,color='blue')
plt.xlabel('h')
plt.ylabel('LL')
plt.show()

h[np.array(ll).argmax()]

#Exercise 15 
#There is no k-nn density estimation library in Python (e.g. like knnDE() in R; package "TDA"); make your own function (next exercise)

#Exercise 16 
def knn(trn,tst,k):
	from scipy.spatial import distance
	testDistMat = distance.cdist(tst,trn,'euclidean')  #need to be arrays
	n = len(trn)
	p = [] 
	for i in range(testDistMat.shape[0]):
		knndist = np.sort(testDistMat[i,:])[k-1] #ascending 
		p.append((k/n)/knndist)
	return p	
	
path_to_file = './data/dataset1D.txt'
data = pd.read_csv(path_to_file, sep=' ')

trn = np.array(data['a']).reshape(-1,1)                #entire dataset
tst = np.arange(trn.min(),trn.max(),0.1).reshape(-1,1) #grid
k = 3

p = knn(trn,tst,k) #inputs must be arrays
plt.plot(tst,p)
plt.xlabel('a')
plt.ylabel('density, k=3')
plt.ylim([-0.2,max(p)])  
plt.scatter(trn,np.zeros(len(trn)),marker='+',color='red')
plt.show()
