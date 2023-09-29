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
from poisson_fourier import DeepONet

# 系统设置
device_cpu = torch.device('cpu')
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 20

# 右端项
def f(sensors):
    f_in = -32 * np.pi**2 * np.sin(4 * np.pi * sensors[:,0]) * np.sin(4 * np.pi * sensors[:,1])
    f_in = np.reshape(f_in, (1,f_in.shape[0]))
    return f_in

# 参数设置
lb_geom = -1 # x下界
ub_geom = 1
dim_s = 64 # 感知点内部维度
dim_y = 500 # y内部训练点维度

# 生成感知点(sensors)
X1 = np.linspace(lb_geom, ub_geom, dim_s)
X2 = np.linspace(lb_geom, ub_geom, dim_s)
X1, X2 = np.meshgrid(X1, X2)
X1 = X1.flatten().reshape(dim_s*dim_s,1)
X2 = X2.flatten().reshape(dim_s*dim_s,1)
sensors = np.c_[X1, X2] 

# 得到测试集
X1 = np.linspace(lb_geom, ub_geom, dim_y)
X2 = np.linspace(lb_geom, ub_geom, dim_y)
X1, X2 = np.meshgrid(X1, X2)
X1 = X1.flatten().reshape(dim_y*dim_y,1)
X2 = X2.flatten().reshape(dim_y*dim_y,1)
ys = np.c_[X1, X2] 
# ys = np.loadtxt(path + "../data/points.txt", dtype = float, delimiter=" ") # 也可以给定测试集，测试集需要是(N,2)维数据
f_in = - f(sensors) # u = Laplace^{-1}(-f)
f_in = torch.from_numpy(f_in).float().to(device_cpu)
X_pred = ys
X_pred = torch.from_numpy(X_pred).float().to(device_cpu)

# 得到预测值并画图
model2 = torch.load(path + '../output/boundary_300000/network.pkl', map_location=device_cpu)
u_pred = model2.predict(f_in, X_pred).flatten()
fig, ax = plt.subplots()
levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
cs = ax.contourf(X1.reshape(dim_y,dim_y), X2.reshape(dim_y,dim_y), u_pred.reshape(dim_y,dim_y), levels,cmap=plt.get_cmap('Spectral'))
cbar = fig.colorbar(cs)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('$u$(pred)')
plt.savefig(path + "u_pred.png", format="png", dpi=300, bbox_inches="tight")