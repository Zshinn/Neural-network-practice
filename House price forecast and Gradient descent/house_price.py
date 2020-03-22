import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MSE(X,y,theta):
    h = np.power(((X * theta.T) - y), 2)
    return np.sum(h)/(2*len(X))

def gradientDescent(X,y,alpha,theta,iters):
    parameter_num=int(theta.ravel().shape[1])
    t=np.matrix(np.zeros(theta.shape))
    loss=np.zeros(iters)
    for i in range(iters):
        # theta的公共导数部分
        delta=(X*theta.T)-y
        # 更新 theta
        for j in range(parameter_num):
            t[0,j]=theta[0,j]-(alpha/len(X))*np.sum(np.multiply(delta,X[:,j]))
        theta=t
        loss[i]=MSE(X,y,theta)
    return theta,loss



path='ex1data2.txt'
data=pd.read_csv(path,names=['Size', 'Bedrooms', 'Price'])
# 归一化
data = (data - data.mean()) / data.std()
data.insert(0,'One',1)
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0,0]))
alpha=0.01
iters=1000
g,loss=gradientDescent(X,y,alpha,theta,iters)
print(MSE(X,y,theta))


#画图
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),loss)

plt.show()


