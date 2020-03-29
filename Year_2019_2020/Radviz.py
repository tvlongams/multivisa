# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:29:00 2017

@author: VTRAN
"""
#import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import requests
from io import StringIO
import numpy as np
import pandas as pd
#from pandas.tools.plotting import radviz
#from pandas.tools.plotting import parallel_coordinates
#from pandas.tools.plotting import scatter_matrix

import time
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import sklearn
from scipy.optimize import differential_evolution
from sklearn.metrics import silhouette_score as silhouette

from sklearn.metrics.pairwise import euclidean_distances

import warnings
warnings.filterwarnings("ignore")

from datasets import *
'''
uci_data=['opt.tes', 'yeast', 'wpdc', 'auto', 'wdbc', 'olive',
'ecoli', 'y14c', 'pen.tes', 'opt.tra', 'bcw', 'pen.tra', 'wine', 'iris']
'''
(x,y)=DATA['iris']
#(x,y)=DATA['wine']
#(x,y)=DATA['y14c']
#(x,y)=DATA['ecoli']
#(x,y)=DATA['olive']
#(x,y)=DATA['auto']


#(x,y)=DATA['bcw']
#(x,y)=DATA['wdbc']
#(x,y)=DATA['wpdc'] 
#(x,y)=DATA['yeast']
#(x,y)=DATA['pen.tra']
#(x,y)=DATA['pen.tes']
#(x,y)=DATA['opt.tra']
#(x,y)=DATA['opt.tes']

#from readdata import *
'''
istar_data=['All_reduced.DATA',  'dermatology.data', 'elephant.DATA', 'ETHZ.data', 'fiber-notnorm.data', 
'freeFoto.DATA', 'mammal.data', 'Mice.data', 'movementLibras.data', 'optdigits.DATA', 
'PHYSHING.DATA', 'QSAR.data', 'satimage.data', 'segment.data', 'SegmentationNormcols.DATA', 
'shapes.data', 'SpamBase.data', 'texture.data', 'twonorm.data', 'VEHICLE.DATA', 
'wdbc_std.data','ionosphere.txt', 'primary-tumor.txt', 'SONAR.txt', 'spectfheart.txt']
'''
#(x,y)=DT[2]



def normalize_cols(m):
    col_max=m.max(axis=0)
    col_min=m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

def ClDM(x,y):
	#cldm=1/K*sum d^2(i,j)/(r_i*r_j)
	cldm=0.0
	data=pd.DataFrame(x)
	data['G']=y
	mean=data.groupby(['G']).mean()
	dist=euclidean_distances(mean,mean)
	#mean.index Int64Index([1, 2, 3], dtype='int64', name='G')
	#mean.iloc[i]
	c=np.unique(data['G'])
	K=len(c)
	r=np.zeros(K)
	#m=mean[mean.index==c[i]].values[0] # mean of group ith
	for i in range(K):
		a1=data[data['G']==c[i]]
		a1=a1.drop(['G'],axis=1).head()
		b1=a1.mean()
		r1=euclidean_distances(a1,b1).mean()
		r[i]=r1
	#a1.drop(['group'],axis=1).head()
	
	s=0.0
	for i in range(K):
		for j in range(i+1,K):
			s=s+np.power(dist[i,j],2)/(r[i]*r[j])
	cldm=s/K
	
	return cldm

def NearestCentroidClassifier(x,y):
	#'''
	#clf=NearestCentroid()
	#clf=LDA()
	#'''
	k=np.int(np.sqrt(len(x)))
	clf=KNN(n_neighbors=k)
	clf.fit(x,y)
	ac=clf.score(x,y)
	#'''
	
	#ac=ClDM(x,y)
	#ac=silhouette(x, y, metric='sqeuclidean')
	return ac


	
def StarCoordinates(m):
	theta=np.pi*2/m
	v=np.zeros((m,2))
	for i in range(m):
		v[i,0]=np.cos(i*theta)
		v[i,1]=np.sin(i*theta)
	return v

X=np.nan_to_num(normalize_cols(x))
(n,m)=X.shape

# circular Radviz
def cradviz(x,alpha):
	#alpha
	m=len(x)
	theta=np.zeros(m)
	
	for i in range(m):
		theta[i]=alpha[i]#*x[i]+alpha[i,1]*(1-x[i])
		
	anchor=np.array([np.cos(theta),np.sin(theta)])
	p=np.dot(anchor,x)/np.sum(x)
	return p


def circleradviz(X,a):
	#p=sum 
	(n,m)=X.shape
	Y=np.zeros((n,2))
	for i in range(n):
		Y[i]=cradviz(X[i],a)
	return Y
	
#Differential evolution
bounds = [(0.0,2.0*np.pi)]*(m)
def func(alpha):
	#m=np.int(len(alpha))
	v=alpha#.reshape((m,2))
	#v=x
	#Y=np.dot(X,v)
	Y=circleradviz(X,v)
	
	ac=1.0-NearestCentroidClassifier(Y,y)
	#ac=1.0-silhouette(Y,y)
	return ac
	
'''The differential evolution strategy to use. Should be one of:

‘best1bin’
‘best1exp’
‘rand1exp’
‘randtobest1exp’
‘best2exp’
‘rand2exp’
‘randtobest1bin’
‘best2bin’
‘rand2bin’
‘rand1bin’
'''
'''
Finds the global minimum of a multivariate function. 
Differential Evolution is stochastic in nature (does not use gradient methods) 
to find the minimium, and can search large areas of candidate space, 
but often requires larger numbers of function evaluations 
than conventional gradient based techniques.
'''
#result = differential_evolution(func, bounds,strategy='best1bin',maxiter=50,popsize=30)

#NP = 75, CR = 0.8803, F = 0.4717

#'''
result = differential_evolution(func, bounds,strategy='rand1bin',
			maxiter=50,popsize=15,mutation=0.4717,recombination=0.8803)

print("Radviz: KNN Classifier: ")
print(1.0-result.fun)

x_optimal=result.x
#'''

'''
#original position 
theta=np.zeros(m)
for i in range(m):
	theta[i]=2*i*np.pi/m
	
x_optimal=theta
'''

V=x_optimal#.reshape((m,2))
#Y=np.dot(X,V_optimal)
Y=circleradviz(X,V)
'''
idx=np.argsort(V)


'''
print("optimal: ",V)

classes=np.unique(y)
colors=matplotlib.pyplot.cm.gist_rainbow(np.linspace(0,1,len(classes)))
#colors=matplotlib.pyplot.cm.rainbow(np.linspace(0,1,len(classes)))
cm = matplotlib.colors.ListedColormap(colors)
axiscolors=matplotlib.pyplot.cm.rainbow(np.linspace(0,1,m))

fig=plt.figure(figsize=(6,6))
#draw anchor segment point

pm=np.argsort(V)

for i in range(m):
	t = V[i]#np.linspace(V[i,0],V[i,1], 30)
	plt.scatter(np.cos(t),np.sin(t),color=axiscolors[pm[i]],alpha=0.95)
	plt.text((1+0.05)*np.cos(t),(1+0.05)*np.sin(t),str(pm[i]+1),color="black")

#Anchor=np.array([np.cos(t),np.sin(t)])
#plt.scatter(Anchor[0,:],Anchor[1,:],color="red",s=30*V_optimal)

#draw unit circle
t = np.linspace(0,2*np.pi, 100)
plt.plot(np.cos(t),np.sin(t),alpha=0.95)

plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)

plt.scatter(Y[:,0],Y[:,1],c=y,cmap=cm,marker='.',s=10)

#plt.axis('off')
#ax = plt.subplot(111,aspect = 'equal')

#fig.subplots_adjust(left=0.1, bottom=0.1, right=0.1, top=0.1, wspace=0, hspace=0)

#plt.tight_layout()
#print(ClDM(Y,y))

plt.savefig("KNN\RadvizKNN.png"
	#, bbox_inches='tight'
	#,pad_inches=0.1
	)

plt.show()
	



