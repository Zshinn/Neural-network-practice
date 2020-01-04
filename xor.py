import torch
import numpy as np
from torch.autograd import Variable
np.set_printoptions(threshold=np.inf)

# prepare for data
# 0-7的抑或两两组合 共有36种组合
x=np.zeros([36,2])
y=np.zeros([36,1])
a=0
for i in range(0,8):
    for j in range(i,8):
        x[a][0]=i
        x[a][1]=j
        c=i^j
        y[a][0]=int(c)
        a+=1

input_x=Variable(torch.Tensor(x))
input_x=input_x.float()
y=Variable(torch.from_numpy(y))
y=y.float()

# set network
class network(torch.nn.Module):
    def __init__(self,in_num,hidden_num,out_num):
        super(network,self).__init__()
        self.input_layer=torch.nn.Linear(in_num,hidden_num)
        self.relu=torch.nn.ReLU()
        self.output_layer=torch.nn.Linear(hidden_num,out_num)
        self.softmax=torch.nn.LogSoftmax()
    def forward(self,input_x):
        h_1 = self.relu(self.input_layer(input_x))
        h_2 = self.relu(self.output_layer(h_1))
        return h_2

# train
net=network(2,6,1)
loss_function=torch.nn.MSELoss()
#loss_function=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
for i in range(10000):
    out=net(input_x)
    loss=loss_function(out,y)
    print("loss is %f %f" % (loss.data.numpy(), i))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
out=net(input_x)
print(out)
print(y)