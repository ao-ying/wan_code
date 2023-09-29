import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import time
import torch
import torch.nn as nn         
import torch.optim as optim    
import copy         
from torch.nn.parameter import Parameter
from poisson_2d_sep import PINN, Residual
from poisson_pretrain import PINN_Main

# 系统设置
device_cpu = torch.device('cpu')
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 20

# 参数设置
lb_geom = -1 # x下界
ub_geom = 1
dim = 500

# 得到测试集
X1 = np.linspace(lb_geom, ub_geom, dim)
X2 = np.linspace(lb_geom, ub_geom, dim)
X1, X2 = np.meshgrid(X1, X2)
X1 = X1.flatten().reshape(dim*dim,1)
X2 = X2.flatten().reshape(dim*dim,1)
points = np.c_[X1, X2] # N * 2
tool = np.full((X1.shape[0], 8), 1/8) # 使更高维数值上均为1
points = np.c_[points, tool] # N * dim_x，N个测试点
# points = np.loadtxt(path + "../data/points.txt", dtype = float, delimiter=" ") # 也可以给定测试集，测试集需要是(N,10)维数据
X_pred = points
X_pred = torch.from_numpy(X_pred).float().to(device_cpu)

# 得到预测值并画图
model2 = torch.load(path + '../output/3*3*64_boundaryw3_1000/network.pkl', map_location=device_cpu)
u_pred = model2.predict(X_pred).flatten()
fig, ax = plt.subplots()
levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u_pred.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
cbar = fig.colorbar(cs)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('$u$(pred)')
plt.savefig(path + "u_pred.png", format="png", dpi=300, bbox_inches="tight")