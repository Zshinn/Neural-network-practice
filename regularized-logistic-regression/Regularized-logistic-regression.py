
# 设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果。
# 对于这两次测试，你想决定是否芯片要被接受或抛弃。
# 为了帮助你做出艰难的决定，你拥有过去芯片的测试数据集，从其中你可以构建一个逻辑回归模型。

import pandas as pd
import numpy as np
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1+ np.exp(-z))

def cost(theta,X,y,lambd):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)

    t1=np.multiply(-y,np.log(sigmoid(X*theta.T)))
    t2=np.multiply(1-y,np.log(1-sigmoid(X*theta.T)))
    t3=lambd/(2*len(X)) * np.sum(np.power(theta[:,1:-1],2))

    return 1/len(X) * np.sum(t1-t2) + t3

def gradient(theta,X,y,lambd):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    nums=int(theta.ravel().shape[1])
    grad=np.zeros(nums)

    t=sigmoid(X*theta.T)-y
    for i in range(nums):
        tm=np.multiply(t,X[:,i])
        if(i==0):
            grad[i]=np.sum(tm)/len(X)
        else:
            grad[i]=np.sum(tm)/len(X) + (lambd/len(X))*theta[:,i]

    return grad



data=pd.read_csv('ex2data2.txt',names=['test 1','test 2','ac'])

x1=data['test 1']
x2=data['test 2']

data.insert(3,'one',1)

# 创建一组多项式特征
d=5
for i in range(1,d):
    for j in range(0,i):
        data['F'+str(i)+str(j)]=np.power(x1,i-j)*np.power(x2,j)

data.drop('test 1',axis=1,inplace=True)
data.drop('test 2',axis=1,inplace=True)

cols=data.shape[1]
X=data.iloc[:,1:cols]
y=data.iloc[:,0:1]

X=np.array(X)
y=np.array(y)
theta=np.zeros(11)

result=opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,y,1))

print(result)
