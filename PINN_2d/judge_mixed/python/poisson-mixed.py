import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import time
import torch
import torch.nn as nn         
import torch.optim as optim             
from torch.nn.parameter import Parameter

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
    
#  将优化器optimizer的学习率设置为lr
def reset_lr(optimizer, lr):
    for params in optimizer.param_groups: 
            params['lr'] = lr

# 生成数据集
# lb_x：x中每个元素的下界，ub_x：x中每个元素的上界，dim：x的维度，num_interior求解域内采样点个数，num_boundary边界上采样点个数(每个维度的边界采num_boundary/dim个点，上下边界各一半)
def load_data(lb_geom, ub_geom, num_interior, num_boundary, dim_x):
    # 训练集
    ## 区域内
    X_train = np.random.uniform(lb_geom, ub_geom, (num_interior, dim_x))
    ## 边界
    num_per_dim = int(num_boundary / dim_x)
    tool = np.zeros(num_per_dim)
    tool[:int(num_per_dim/2)] = ub_geom
    tool[int(num_per_dim/2):] = lb_geom
    for i in range(dim_x):
        boundary_points = np.random.uniform(lb_geom, ub_geom, (num_per_dim, dim_x))
        boundary_points[:, i] = tool
        X_train = np.r_[X_train, boundary_points]
    
    return X_train

# 残差块
class Residual(nn.Module):
    def __init__(self, Layers):
        super().__init__()
        # 数据初始化
        self.data_dim = Layers[0] # 全连接神经网络的宽度，每个全连接层宽度一样，且等于输入输出数据的维度
        self.num_fc = len(Layers) # 全连接神经网络层数
        # 权重初始化
        self.params = []
        self.weights = []
        self.biases = []
        for l in range(0,self.num_fc):
            tempw = torch.zeros(self.data_dim, self.data_dim).to(device)
            w = Parameter(tempw, requires_grad=True)
            nn.init.xavier_uniform_(w, gain=1) 
            tempb = torch.zeros(1, self.data_dim).to(device)
            b = Parameter(tempb, requires_grad=True)
            self.weights.append(w)
            self.biases.append(b) 
            self.params.append(w)
            self.params.append(b)
        
    def forward(self, X):
        H = X
        for l in range(0, self.num_fc):
            W = self.weights[l]
            b = self.biases[l]
            H = torch.add(torch.matmul(H, W), b)
            if l != self.num_fc - 1: # 若不是最后一个全连接层则有激活函数
                H = torch.sin(H) 
        Y = torch.sin(torch.add(H, X)) # 残差连接
        
        return Y

# PINN神经网络
class PINN(nn.Module):
    def __init__(self, num_interior, num_boundary, boundary_weight, lb_X, ub_X, Layers_sep, num_res_sep, Layers_unsep, num_res_unsep, dim_x, dim_u):
        super(PINN, self).__init__()
        # 初始化参数
        self.lb = lb_X
        self.ub = ub_X
        self.Layers_sep = Layers_sep
        self.num_res_sep = num_res_sep
        self.Layers_unsep = Layers_unsep
        self.num_res_unsep = num_res_unsep
        self.num_interior = num_interior
        self.boundary_weight = boundary_weight
        self.params = []
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_res_fc_sep = Layers_sep[0] # 残差块中全连接网络宽度，每层宽度都相等
        self.dim_res_fc_unsep = Layers_unsep[0]
        self.num_boundary = num_boundary
        self.initial_NN()
        
    def initial_NN(self):
        # 可分离变量部分
        ## 初始化第一个全连接层
        self.fc_first_sep = nn.ModuleList([nn.Linear(1, self.dim_res_fc_sep).to(device) for i in range(self.dim_x)])
        for i in range(self.dim_x):
            nn.init.xavier_uniform_(self.fc_first_sep[i].weight, gain=1)
            nn.init.zeros_(self.fc_first_sep[i].bias)
            self.params.append(self.fc_first_sep[i].weight)
            self.params.append(self.fc_first_sep[i].bias)
        ## 初始化残差层
        self.res_blocks_sep = nn.ModuleList([nn.ModuleList([Residual(self.Layers_sep) for j in range(self.num_res_sep)]) for i in range(self.dim_x)])
        for i in range(self.dim_x):
            for j in range(self.num_res_sep):
                self.params.extend(self.res_blocks_sep[i][j].params)
        ## 初始化最后一个全连接层
        self.fc_last_sep = nn.ModuleList([nn.Linear(self.dim_res_fc_sep, self.dim_u).to(device) for i in range(self.dim_x)])
        for i in range(self.dim_x):
            nn.init.xavier_uniform_(self.fc_last_sep[i].weight, gain=1)
            nn.init.zeros_(self.fc_last_sep[i].bias)
            self.params.append(self.fc_last_sep[i].weight)
            self.params.append(self.fc_last_sep[i].bias)
        
        # 不可分离变量部分
        # 初始化第一个全连接层
        self.fc_first_unsep = nn.Linear(dim_x, self.dim_res_fc_unsep)
        nn.init.xavier_uniform_(self.fc_first_unsep.weight, gain=1)
        nn.init.zeros_(self.fc_first_unsep.bias)
        self.params.append(self.fc_first_unsep.weight)
        self.params.append(self.fc_first_unsep.bias)
        # 初始化残差层
        self.res_blocks_unsep = nn.ModuleList([Residual(Layers_unsep) for i in range(self.num_res_unsep)])
        for i in range(self.num_res_unsep):
            self.params.extend(self.res_blocks_unsep[i].params)
        # 初始化最后一个全连接层
        self.fc_last_unsep = nn.Linear(self.dim_res_fc_unsep, self.dim_u)
        nn.init.xavier_uniform_(self.fc_last_unsep.weight, gain=1)
        nn.init.zeros_(self.fc_last_unsep.bias)
        self.params.append(self.fc_last_unsep.weight)
        self.params.append(self.fc_last_unsep.bias)
    
    # 全连接神经网络部分
    def neural_net(self, X):
        # 数据预处理，这里训练和测试时用的lb和ub应该是一样的，否则训练和测试用的神经网络就不一样了。 
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        In = H.float()
        
        # 可分离部分
        ## ResNet部分
        out_list = []
        for i in range(self.dim_x):
            H = self.fc_first_sep[i](In[:,i:(i+1)])
            for j in range(self.num_res_sep):
                H = self.res_blocks_sep[i][j](H)
            Y = self.fc_last_sep[i](H)
            out_list.append(Y)
        ## 各个x分量计算结果相乘得到最终的输出结果
        u1 = out_list[0]
        for i in range(1,self.dim_x):
            u1 = u1 * out_list[i]
            
        # 不可分离部分
        H = self.fc_first_unsep(In)
        for i in range(self.num_res_unsep):
            H = self.res_blocks_unsep[i](H)
        u2 = self.fc_last_unsep(H)
        
        # 总位移
        u = u1 + u2
            
        return u, u1, u2

    # PDE部分
    def he_net(self, X):
        # 方程
        X_e = [0 for i in range(self.dim_x)]
        for i in range(self.dim_x):
            X_e[i] = X[0:self.num_interior, i : i + 1].clone()
            X_e[i] = X_e[i].requires_grad_()
        u_e,_,_ = self.neural_net(torch.cat(X_e, dim = 1))
        dudx = [grad(u_e, X_e[i]) for i in range(self.dim_x)]
        dudx2 = [grad(dudx[i], X_e[i]) for i in range(self.dim_x)]
        dudx2 = torch.cat(dudx2, dim=1) # self.num_interior * dim_x
        Laplace_u = torch.sum(dudx2, dim=1, keepdim=True)  # self.num_interior * 1
        ## 连乘项
        tool = torch.sin(4 * np.pi * X_e[0])
        for i in range(1,self.dim_x):
            tool *= torch.sin(4 * np.pi * X_e[i])
        f = - 16 * self.dim_x * (np.pi ** 2) * tool # 函数f
        equation = - Laplace_u - f

        # 边界条件
        X_b = X[self.num_interior:, :]
        u_b,_,_ = self.neural_net(X_b) # self.num_boundary * self.dim_x
        boundary = u_b - 0

        return equation, boundary

    # 损失函数
    def loss(self,X_train):
        # 计算方程和边界条件项
        equation, boundary = self.he_net(X_train)
        
        # 计算总误差
        loss_e = torch.mean(torch.square(equation))
        loss_b = torch.mean(torch.square(boundary))
        loss_all = loss_e + self.boundary_weight * loss_b

        return loss_all
    
    # 预测X对应点处的u值
    def predict(self, X):
        u_pred,u1_pred,u2_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        u1_pred = u1_pred.cpu().detach().numpy()
        u2_pred = u2_pred.cpu().detach().numpy()
        return u_pred, u1_pred, u2_pred

# main函数
if __name__ == "__main__":
    # 参数设置
    ## 方程相关
    lb_geom = -1 # x下界
    ub_geom = 1 # x上界
    dim_x = 2 # x维度
    dim_u = 1 # u维度
    
    ## 神经网络相关
    name = "5*50_dim2_mixed"
    num_interior = 20000
    num_boundary = 8000 # 必须整除dim_x
    epochs = 20000 # adam优化器迭代次数
    Layers_sep = [50, 50, 50, 50, 50] # 可分离变量部分残差块中全连接网络结构
    num_res_sep = 2 # 可分离变量部分每个通道残差块个数
    Layers_unsep = [50, 50, 50, 50, 50] # 不可分离变量部分残差块中全连接网络结构
    num_res_unsep = 4 # 不可分离变量部分残差块个数
    learning_rate = 0.001 # 初始学习率 
    patience = max(10, epochs / 10) # 平原衰减学习率的patience个数
    boundary_weight = 1 # 设置太大可能导致梯度爆炸
    h = int(1000) # 学习率衰减相关
    r = 0.95 # 学习率每隔h轮迭代乘r
    weight_decay = 0.001 # L2正则项系数，防止损失值突增
    patience = max(10, epochs/10)
    
    ## 画图相关
    train = False
    align = True
    lb_loss = 1e-3
    ub_loss = 1e21
    lb_u = -1.2
    ub_u = 1.2
    
    # 辅助变量
    name = name + ("_%d" % epochs)
    print("\n\n***** name = %s *****" % name)
    output_path = path + '../output/%s/' % name
    if not os.path.exists(output_path): os.mkdir(output_path)

    # 生成数据集
    X_train = load_data(lb_geom, ub_geom, num_interior, num_boundary, dim_x)
    lb_X = X_train.min(0) # 得到训练集每个维度上的最大值组成的元素
    ub_X = X_train.max(0) # 得到训练集每个维度上的最小值组成的元素
    X_train = torch.from_numpy(X_train).float().to(device)
    lb_X = torch.from_numpy(lb_X).float().to(device)
    ub_X = torch.from_numpy(ub_X).float().to(device)
    if train:
        print("Loading data")
        X_train = load_data(lb_geom, ub_geom, num_interior, num_boundary, dim_x)
        lb_X = X_train.min(0) # 得到训练集每个维度上的最大值组成的元素
        ub_X = X_train.max(0) # 得到训练集每个维度上的最小值组成的元素
        X_train = torch.from_numpy(X_train).float().to(device)
        lb_X = torch.from_numpy(lb_X).float().to(device)
        ub_X = torch.from_numpy(ub_X).float().to(device)

        # 声明神经网络实例
        model = PINN(num_interior, num_boundary, boundary_weight, lb_X, ub_X, Layers_sep, num_res_sep, Layers_unsep, num_res_unsep, dim_x, dim_u)
        model = nn.DataParallel(model)
        model = model.module
        model.to(device)
        print(model) # 打印网络概要
        
        # Adam优化器
        optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6)

        # 训练
        start = time.time()
        ## adam
        loss_list = []
        error_list = [] # 保存在测试集上的平均相对误差值
        min_loss = 999999999999
        print("Start training!")
        for it in range(epochs):
            # 优化器训练
            Loss = model.loss(X_train)
            optimizer.zero_grad(set_to_none=True)
            Loss.backward() 
            optimizer.step()

            # 指数衰减学习率
            scheduler.step(Loss)
            # if (it + 1) % h == 0: # 指数衰减
            #     learning_rate *= r
            #     reset_lr(optimizer, learning_rate)

            # 保存损失值和相对误差
            loss_val = Loss.cpu().detach().numpy()
            loss_list.append(loss_val)
            
            # 保存训练误差最小的模型
            if loss_val < min_loss:
                torch.save(model, output_path + 'network.pkl') 
                min_loss = loss_val

            # 输出
            if (it + 1) % (epochs/20) == 0:
                print("It = %d, loss = " % (it + 1), loss_val, ", finish: %d%%" % ((it + 1) / epochs * 100))
            
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
        ## 保存模型参数
        torch.save(model.state_dict(), output_path + 'model.pth')

    # 保存loss曲线
    loss_list = np.loadtxt(output_path + "/loss.txt", dtype = float, delimiter=' ')
    plt.semilogy(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if align:
        plt.ylim(lb_loss, ub_loss) # 与BiPINN统一量纲
        plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=300, bbox_inches="tight")

    # 画图数据准备
    dim = 500
    X1 = np.linspace(lb_geom, ub_geom, dim)
    X2 = np.linspace(lb_geom, ub_geom, dim)
    X1, X2 = np.meshgrid(X1, X2)
    X1 = X1.flatten().reshape(dim*dim,1)
    X2 = X2.flatten().reshape(dim*dim,1)
    if dim_x == 2:
        points = np.c_[X1, X2]
    if dim_x == 10:
        points = np.full((X1.shape[0], dim_x), 1/8) # 使更高维数值上均为1
        points[:,0:1] = X1
        points[:,1:2] = X2
    u_truth = solution(points)
    X_pred = points
    X_pred = torch.from_numpy(X_pred).float().to(device)
    model2 = torch.load(output_path + 'network.pkl', map_location=device)
    u_pred, u1_pred, u2_pred = model2.predict(X_pred)
    u_pred = u_pred.flatten()
    u1_pred = u1_pred.flatten()
    u2_pred = u2_pred.flatten()
    u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
    print("Relative error of u is %.8f when drawing" % u_error)
    np.savetxt(output_path + "u_truth.txt", u_truth, fmt="%s", delimiter=' ')
    np.savetxt(output_path + "u_pred.txt", u_pred, fmt="%s", delimiter=' ')
    np.savetxt(output_path + "u1_pred.txt", u1_pred, fmt="%s", delimiter=' ')
    np.savetxt(output_path + "u2_pred.txt", u2_pred, fmt="%s", delimiter=' ')

    # 画u图像
    fig, ax = plt.subplots()
    if align:
            levels = np.arange(lb_u, ub_u, (ub_u - lb_u) / 100)
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
        
    # 画u1图像
    fig, ax = plt.subplots()
    if align:
            levels = np.arange(lb_u, ub_u, (ub_u - lb_u) / 100)
    else:
        levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
    cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u1_pred.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
    cbar = fig.colorbar(cs)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('$u_1$(pred)')
    if align:
        plt.savefig(output_path + "u1_pred_aligned.png", format="png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(output_path + "u1_pred.png", format="png", dpi=300, bbox_inches="tight")
        
    # 画u2图像
    fig, ax = plt.subplots()
    if align:
            levels = np.arange(lb_u, ub_u, (ub_u - lb_u) / 100)
    else:
        levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
    cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u2_pred.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
    cbar = fig.colorbar(cs)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('$u_2$(pred)')
    if align:
        plt.savefig(output_path + "u2_pred_aligned.png", format="png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(output_path + "u2_pred.png", format="png", dpi=300, bbox_inches="tight")
    
    # 画精确解图像
    fig, ax = plt.subplots()
    if align:
            levels = np.arange(lb_u, ub_u, (ub_u - lb_u) / 100)
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
        
    # 计算标准误差(10维时占用内存较多，建议放在CPU运算)
    if dim_x == 2:
        dim = 100
        X1 = np.linspace(lb_geom, ub_geom, dim)
        X2 = np.linspace(lb_geom, ub_geom, dim)
        X1, X2 = np.meshgrid(X1, X2)
        X1 = X1.flatten().reshape(dim*dim,1)
        X2 = X2.flatten().reshape(dim*dim,1)
        points = np.c_[X1, X2] # N * 2
        u_truth = solution(points)
        X_pred = points
        X_pred = torch.from_numpy(X_pred).float().to(device)
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        u_pred,_,_ = model2.predict(X_pred)
        u_pred = u_pred.flatten()
        u_mae = np.mean(np.abs(u_truth - u_pred))
        print("Mean squared error of u is %.8f" % u_mae)
        u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("Relative error of u is %.8f" % u_error)
    elif dim_x == 10:
        dim = 4
        tool = np.linspace(lb_geom, ub_geom, dim)
        points = [[tool[i]] for i in range(dim)]
        for i in range(dim_x - 1):
            temp = np.array(points)
            for j in range(len(points)):
                for k in range(3):
                    points.append(points[j] + [tool[k]])
                points[j].append(tool[3])
        points = np.array(points)
        u_truth = solution(points)
        X_pred = points
        X_pred = torch.from_numpy(X_pred).float().to(device)
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        u_pred = model2.predict(X_pred).flatten()
        u_mae = np.mean(np.abs(u_truth - u_pred))
        print("Mean squared error of u is %.8f" % u_mae)
        u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("Relative error of u is %.8f" % u_error)