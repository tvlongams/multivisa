import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib

def read_leukemia(file="biolab.si/leukemia.tab"):
    data=pd.read_table(file,delimiter="\t", header=None)
    col=data.columns.values
    gene_name=data.iloc[0][col[1:]]
    y=data[col[0]][3:]
    x=data[col[1:]].iloc[3:].astype(np.float)
    return (x,y)#,gene_name)
	#return (x,y)

def read_colon_cancer():
    data=pd.read_table('COLONCANCER\colon.txt',delimiter=",",header=None)
    col=data.columns.values
    x=data[col[0:len(col)-1]]
    y=data[col[-1]]
    return (x,y)


	
(x,y)=read_leukemia()
x=(x-x.min())/(x.max()-x.min())

def cradviz(x,an):
    #alpha
    #m=len(x)
    #theta=np.zeros(m)
    #for i in range(m):
    #    theta[i]=alpha[i]#*x[i]+alpha[i,1]*(1-x[i])
    anchor=an#p.array([np.cos(theta),np.sin(theta)])
    if np.sum(x)!=0.0:
        p=np.dot(anchor,x)/np.sum(x)
    else:
        p=np.zeros(2)
    return p
	
def circleradviz(X,an):
	#p=sum 
	(n,m)=X.shape
	Y=np.zeros((n,2))
	for i in range(n):
		Y[i]=cradviz(X[i],an)
	return Y
	
def DimensionAnchor(m):
	an=np.zeros(m)
	for i in range(m):
		an[i]=i*np.pi*2/m

	anchor=np.array([np.cos(an),np.sin(an)])
	return anchor
	
def GeneSelection(m,x,y):
	#m=8
	anchor=DimensionAnchor(m)
	best_score=0.0
	select=col[range(m)]
	X=x[select].values
	Y=circleradviz(X,anchor)
	best_score=metrics.silhouette_score(Y, y, metric='sqeuclidean') 
	best_select=select
	# loop
	for i in col[m:]:
		select=best_select.copy()
		for j in range(m):
			c_select=select.copy()
			c_select[j]=i
			X=x[c_select].values
			Y=circleradviz(X,anchor)
			score=metrics.silhouette_score(Y, y, metric='sqeuclidean') 
			if (best_score<score):
				best_score=score
				best_select=c_select
	return (best_score,best_select)
	

	
classes=np.unique(y)
colors=matplotlib.pyplot.cm.rainbow(np.linspace(0,1,len(classes)))
cm = matplotlib.colors.ListedColormap(colors)
col=x.columns.values
cy=pd.factorize(y)[0]

#number of genes
m=6
(score,select)=GeneSelection(m,x,y)

X=x[select].values
anchor=DimensionAnchor(m)

Y=circleradviz(X,anchor)
best_score=metrics.silhouette_score(Y, y, metric='sqeuclidean') 
print('Score: ', best_score)
plt.figure(figsize=(6,6))
cy=pd.factorize(y)[0]

t=np.linspace(0,2*np.pi,100)
plt.plot(np.cos(t),np.sin(t))

plt.scatter(anchor[0,:],anchor[1,:])

plt.scatter(Y[:,0],Y[:,1],c=cy,cmap=cm,marker='.',s=10)

plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.show()