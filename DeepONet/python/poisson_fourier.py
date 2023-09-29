import os
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import time
import torch
import torch.nn as nn 
import torch.optim as optim   
from torch.nn.parameter import Parameter  
from sympy import Symbol, pi, sin, cos, sqrt, Min, Max, Abs, Pow       

# 系统设置
seed = 11
np.random.seed(seed)
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 训练时为"3"，预测时为""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 20

# 精确解
# x待求值点的d维向量，若有n个点待求则x的维度为n*d
def solution(x):
    dim = x.shape[1]
    ret = -1
    for i in range(dim):
        ret *= np.sin(4 * np.pi * x[:,i])
    return ret

# 计算f在感知点处的DeepONet输入信号。
def f(sensors):
    f_in = -32 * np.pi**2 * np.sin(4 * np.pi * sensors[:,0]) * np.sin(4 * np.pi * sensors[:,1])
    f_in = np.reshape(f_in, (1,f_in.shape[0]))
    return f_in

# 求偏导
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# 输出模型参数信息
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    # print("Total: %d, Trainable: %d" % (total_num, trainable_num))
    return trainable_num

# 生成输入函数对应的感知点和y对应的训练点
def generate_points(dim_s, dim_y, num_boundary):
    # 生成感知点(sensors)
    X1 = np.linspace(lb_geom, ub_geom, dim_s)
    X2 = np.linspace(lb_geom, ub_geom, dim_s)
    X1, X2 = np.meshgrid(X1, X2)
    X1 = X1.flatten().reshape(dim_s*dim_s,1)
    X2 = X2.flatten().reshape(dim_s*dim_s,1)
    sensors = np.c_[X1, X2] 
    
    # y对应的训练点
    ## 内部训练点
    X1 = np.linspace(lb_geom, ub_geom, dim_y)
    X2 = np.linspace(lb_geom, ub_geom, dim_y)
    X1, X2 = np.meshgrid(X1, X2)
    X1 = X1.flatten().reshape(dim_y*dim_y,1)
    X2 = X2.flatten().reshape(dim_y*dim_y,1)
    ys = np.c_[X1, X2] # m * 2
    ## 添加边界点
    points = np.linspace([lb_geom,lb_geom],[ub_geom,lb_geom],num_boundary,endpoint=False) # 下
    ys = np.r_[ys, points]
    points = np.linspace([ub_geom,lb_geom],[ub_geom,ub_geom],num_boundary,endpoint=False) # 右
    ys = np.r_[ys, points]
    points = np.linspace([lb_geom,lb_geom],[lb_geom,ub_geom],num_boundary,endpoint=False) # 左
    ys = np.r_[ys, points]
    points = np.linspace([lb_geom,ub_geom],[ub_geom,ub_geom],num_boundary,endpoint=False) # 上
    ys = np.r_[ys, points]
    
    return sensors, ys

# 生成DeepONet训练数据
# sensors为感知点，ys为在求解域内选取的训练点。
def load_data(sensors, ys, dim_base):
    # 函数构造
    m = sensors.shape[0]
    N = ys.shape[0]
    s1 = sensors[:,0]
    s2 = sensors[:,1]
    y1 = ys[:,0]
    y2 = ys[:,1]
    x, y = Symbol('x'), Symbol('y')
    f_x = []
    f_y = []
    for i in range(1, dim_base + 1):
        f_x.append(sin(i * pi * x))
        # f_x.append(cos(i * pi * x))
        f_y.append(sin(i * pi * y))
        # f_y.append(cos(i * pi * y))
    u_in = []
    Y_train = []
    # for i in range(len(f_x)): # 所有基函数
    #     for j in range(len(f_y)):
    #         fun = f_x[i] * f_y[j]
    #         if i == 0 and j == 0:
    #             u_in.append(np.zeros((1,m)))
    #             Y_train.append(np.full((N,1), fun))
    #         else:
    #             f_base = sp.lambdify((x,y), fun, 'numpy')
    #             Laplace_f_base = sp.lambdify((x,y), fun.diff(x,2) + fun.diff(y,2), 'numpy')
    #             u_in.append(Laplace_f_base(s1,s2).reshape(1,m))
    #             Y_train.append(f_base(y1,y2).reshape(N,1))
    for i in range(len(f_x)): # 对角线上的基函数
        j = i
        fun = f_x[i] * f_y[j]
        # if i == 0 and j == 0:
        #     u_in.append(np.zeros((1,m)))
        #     Y_train.append(np.full((N,1), fun))
        # else:
        f_base = sp.lambdify((x,y), fun, 'numpy')
        Laplace_f_base = sp.lambdify((x,y), fun.diff(x,2) + fun.diff(y,2), 'numpy')
        u_in.append(Laplace_f_base(s1,s2).reshape(1,m))
        Y_train.append(f_base(y1,y2).reshape(N,1))
    
    # u
    u_in = np.concatenate(u_in, 0)
    if len(u_in.shape) == 1:
        u_in = np.reshape(u_in, (1,u_in.shape[0]))
    
    # Laplace^{-1}(u)
    Y_train = np.concatenate(Y_train, 1)
    if len(Y_train.shape) == 1:
        Y_train = np.reshape(Y_train, (Y_train.shape[0],1))
      
    return u_in, Y_train

class DeepONet(nn.Module):
    def __init__(self, Layers_branch, Layers_trunk, p, dim_x, dim_u, u_in):
        super(DeepONet, self).__init__()
        self.Layers_branch = [u_in.shape[1]] + Layers_branch + [p]
        self.Layers_trunk = [dim_x] + Layers_trunk + [p]
        self.dim_x = dim_x
        self.p = p
        self.dim_u = dim_u
        self.act = torch.tanh
        self.initial_NN()
        
    def initial_NN(self):
        self.params = []
        # Branch net
        self.branck_net = nn.ModuleList([nn.Linear(self.Layers_branch[i], self.Layers_branch[i+1]) for i in range(len(self.Layers_branch) - 1)])
        for i in range(len(self.Layers_branch) - 1):
            nn.init.xavier_uniform_(self.branck_net[i].weight, gain=1)
            nn.init.zeros_(self.branck_net[i].bias)
            self.params.append(self.branck_net[i].weight)
            self.params.append(self.branck_net[i].bias)
            
        # Trunk net
        self.trunk_net = nn.ModuleList([nn.Linear(self.Layers_trunk[i], self.Layers_trunk[i+1]) for i in range(len(self.Layers_trunk) - 1)])
        for i in range(len(self.Layers_trunk) - 1):
            nn.init.xavier_uniform_(self.trunk_net[i].weight, gain=1)
            nn.init.zeros_(self.trunk_net[i].bias)
            self.params.append(self.trunk_net[i].weight)
            self.params.append(self.trunk_net[i].bias)
            
        # bias
        tempb = torch.tensor(0.0).to(device)
        self.b0 = Parameter(tempb, requires_grad=True)
        self.params.append(self.b0)
            
    def neural_net(self,u_in,ys):
        # Branch net
        B = u_in
        for i in range(len(self.Layers_branch) - 1):
            B = self.branck_net[i](B)
            if i != len(self.Layers_branch) - 2:
                B = self.act(B)
        
        # Trunk net
        T = ys
        for i in range(len(self.Layers_trunk) - 1):
            T = self.trunk_net[i](T)
            if i != len(self.Layers_trunk) - 2:
                T = self.act(T)
                
        # 构建总输出
        out = B[0:1, :] * T
        out = torch.sum(out, dim=1, keepdim=True)
        for i in range(1, B.shape[0]):
            temp = B[i : (i+1), :] * T
            temp = torch.sum(temp, dim=1, keepdim=True)
            out = torch.cat((out, temp), dim=1)
        out = out + self.b0
        
        return out
    
    # 损失函数
    def loss(self,u_in,ys,Y_train):
        # 计算方程和边界条件项
        Y_pred = self.neural_net(u_in,ys)
        
        # 计算总误差
        loss = torch.mean(torch.square(Y_pred - Y_train))

        return loss
    
    # 预测X对应点处的u值
    def predict(self, f_in, ys):
        u_pred = self.neural_net(f_in, ys)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred 

# main函数
if __name__ == "__main__":
    # 参数设置
    ## 方程相关
    lb_geom = -1 # x下界
    ub_geom = 1 # x上界
    dim_x = 2 # x维度
    dim_u = 1 # u维度
    
    ## 神经网络相关
    name = "boundary"
    epochs = 300000
    p = 100
    Layers_branch = [2560, 2560]
    Layers_trunk = [1280, 1280, 1280]
    learning_rate = 0.0001
    dim_s = 64 # 感知点内部维度
    dim_y = 500 # y训练点维度
    num_boundary = 80 # 每条边上选择的训练点数
    dim_base = 5 # 基为1,sin(pi * x),cos(pi * x),...,sin(dim_base * pi * x),cos(dim_base * pi * x)拓展至二维
    patience = max(10, epochs/10)
    
    ## 画图相关
    train = False
    align = True
    lb_loss = 1e-3
    ub_loss = 1e21
    lb_u = -1.1
    ub_u = 1.1
    
    # 辅助变量
    name = name + ("_%d" % epochs)
    print("\n\n***** name = %s *****" % name)
    output_path = path + '../output/'
    if not os.path.exists(output_path): os.mkdir(output_path)
    output_path = path + '../output/%s/' % name
    if not os.path.exists(output_path): os.mkdir(output_path)
    
    # 数据集生成
    sensors, ys = generate_points(dim_s, dim_y, num_boundary)
    
    # 误差测试数据生成
    dim = 256
    X1 = np.linspace(lb_geom, ub_geom, dim)
    X2 = np.linspace(lb_geom, ub_geom, dim)
    X1, X2 = np.meshgrid(X1, X2)
    X1 = X1.flatten().reshape(dim*dim,1)
    X2 = X2.flatten().reshape(dim*dim,1)
    points = np.c_[X1, X2] # N * 2
    u_truth = solution(points)
    X_pred = points
    X_pred = torch.from_numpy(X_pred).float().to(device)
    f_in = - f(sensors) # u = Laplace^{-1}(-f)
    f_in = torch.from_numpy(f_in).float().to(device)
    if train:
        # 生成训练数据
        print("Loading data")
        u_in, Y_train = load_data(sensors, ys, dim_base)
        u_in = torch.from_numpy(u_in).float().to(device)
        ys = torch.from_numpy(ys).float().to(device)
        Y_train = torch.from_numpy(Y_train).float().to(device)
        
        # 声明神经网络实例
        model = DeepONet(Layers_branch, Layers_trunk, p, dim_x, dim_u, u_in)
        model = nn.DataParallel(model)
        model = model.module
        model.to(device)
        print(model) # 打印网络概要
        
        # Adam优化器
        optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6)
        
        # 训练
        start = time.time()
        loss_list = []
        min_loss = 999999999999
        print("Start training!")
        for it in range(epochs):
            # 优化器训练
            Loss = model.loss(u_in,ys,Y_train)
            optimizer.zero_grad(set_to_none=True)
            Loss.backward() 
            optimizer.step()

            # 衰减学习率
            scheduler.step(Loss)
            
            # 保存损失值和相对误差
            loss_val = Loss.cpu().detach().numpy()
            loss_list.append(loss_val)
            
            # 保存训练误差最小的模型
            if loss_val < min_loss:
                    torch.save(model, output_path + 'network.pkl') 
                    min_loss = loss_val

            # 输出
            if (it + 1) % (epochs/20) == 0:
                # 计算相对误差
                model2 = torch.load(output_path + 'network.pkl', map_location=device) 
                u_pred = model2.predict(f_in, X_pred).flatten()
                u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
                # 输出
                print("It = %d, loss = %.8f, u_error = %.8f, finish: %d%%" % ((it + 1), loss_val, u_error, ((it + 1) / epochs * 100)))
                
        ## 后续处理
        end = time.time()
        train_time = end - start
        loss_list = np.array(loss_list).flatten()
        min_loss = np.min(loss_list)
        np.savetxt(output_path + "/loss.txt", loss_list, fmt="%s",delimiter=' ')
        params = get_parameter_number(model)
        print("time = %.2fs" % train_time)
        print("params = %d" % params)
        print("min_loss = %.8f" % min_loss)
    
    # 保存loss曲线
    loss_list = np.loadtxt(output_path + "/loss.txt", dtype = float, delimiter=' ')
    plt.semilogy(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # if align:
    #     plt.ylim(lb_loss, ub_loss) # 与BiPINN统一量纲
    #     plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
    # else:
    plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=300, bbox_inches="tight")
    
    # 画图数据准备
    model2 = torch.load(output_path + 'network.pkl', map_location=device) 
    u_pred = model2.predict(f_in, X_pred).flatten()
    u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
    print("Final: Relative error of u is %.8f" % u_error)
    np.savetxt(output_path + "u_truth.txt", u_truth, fmt="%s", delimiter=' ')
    np.savetxt(output_path + "u_pred.txt", u_pred, fmt="%s", delimiter=' ')
    
    # 画预测解图像
    fig, ax = plt.subplots()
    if align:
            levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
    else:
        levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
    cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u_pred.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
    cbar = fig.colorbar(cs)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('$u$(pred)')
    if align:
        plt.savefig(output_path + "u_pred_aligned.png", format="png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(output_path + "u_pred.png", format="png", dpi=300, bbox_inches="tight")
    
    # 画精确解图像
    fig, ax = plt.subplots()
    if align:
            levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
    else:
        levels = np.arange(min(u_truth) - abs(max(u_truth) - min(u_truth)) / 10, max(u_truth) + abs(max(u_truth) - min(u_truth)) / 10, (max(u_truth) - min(u_truth)) / 100) 
    cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u_truth.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
    cbar = fig.colorbar(cs)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('$u$(truth)')
    if align:
        plt.savefig(output_path + "u_truth_aligned.png", format="png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(output_path + "u_truth.png", format="png", dpi=300, bbox_inches="tight")
    