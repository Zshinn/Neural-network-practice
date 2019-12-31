import torch
import numpy as np
from torch.autograd import Variable
np.set_printoptions(threshold=np.inf)

# prepare for data
# 0-7的抑或两两组合 共有36种组合
x=np.zeros([36,2])
yt=np.zeros([36],dtype=int)
y=np.zeros([36,8])
a=0
for i in range(0,8):
    for j in range(i,8):
        x[a][0]=i
        x[a][1]=j
        c=i^j
        yt[a]=int(c)
        a+=1
# 将yT的结果转换成二进制
for i in range(36):
    a = bin(yt[i]).replace('0b', '')
    for j in range(len(a)):
        y[i][7-j]=float(a[len(a)-1-j])

input_x=Variable(torch.Tensor(x))
input_x=input_x.float()
y=Variable(torch.from_numpy(y))
y=y.float()

# set network
class network(torch.nn.Module):
    def __init__(self,in_num,hidden_num,out_num):
        super(network,self).__init__()
        self.input_layer=torch.nn.Linear(in_num,hidden_num)
        self.sigmoid=torch.nn.Sigmoid()
        self.output_layer=torch.nn.Linear(hidden_num,out_num)
        self.softmax=torch.nn.LogSoftmax()
    def forward(self,input_x):
        h_1 = self.sigmoid(self.input_layer(input_x))
        h_2 = self.sigmoid(self.output_layer(h_1))
        return h_2

# train
net=network(2,6,8)
loss_function=torch.nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
for i in range(10000):
    out=net(input_x)
    loss=loss_function(out,y)
    print ("loss is %f"%loss.data.numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
out=net(input_x)
print(out)
print(y)