# 代码说明

# 1、代码结构

```python
PINN_2d # PINN求解二维泊松方程
  judge_mixed #计算机方法判断变量分离
    python # 训练及预测源代码
    output  # 训练好的模型以及结果图
  judge_base_function # 基函数方法判断变量分离
    python
    output
  separate # 变量可分离神经网络
    python
    output
PINN_10d # PINN求解十维泊松方程
  judge_mixed # 计算机方法判断变量分离
    python
    output
  separate # 变量可分离神经网络
    python
    output
DeepONet # DeepONet求解二维泊松方程
  data
  python
  output
```



# 2、环境说明

```python
python 3.9.13
torch 2.0.1
numpy 1.24.1
matplotlib 3.4.3
CUDA 12.2
GPU: NVIDIA GeForce RTX 3090
```



# 3、得到任意点处的预测值

## 3.1、PINN求解二维泊松方程

运行`code/PINN_2d/separate/predict.py`，默认预测竞赛给定的测试集并画图，同时将预测值存为`code/PINN_2d/separate/u_pred.txt`文件。

如果用文件的形式给定预测点，请将预测点文件放在`code/PINN_2d/separate/data`文件夹中并命名为`points.txt`，预测点文件应该是`(N,2)`维数组，其中`N`是预测点的个数。并将`code/PINN_2d/separate/predict.py`中的参数`is_file`设置为`True`，然后运行文件即可得到指定点处的值，同时将预测值存为`code/PINN_2d/separate/u_pred.txt`文件。

 

## 3.2、PINN求解十维泊松方程

运行`code/PINN_10d/separate/predict.py`，默认预测竞赛给定的测试集并画图，同时将预测值存为`code/PINN_10d/separate/u_pred.txt`文件。

如果用文件的形式给定预测点，请将预测点文件放在`code/PINN_10d/separate/data`文件夹中并命名为`points.txt`，预测点文件应该是`(N,10)`维数组，其中`N`是预测点的个数。并将`code/PINN_10d/separate/predict.py`中的参数`is_file`设置为`True`，然后运行文件即可得到指定点处的值，同时将预测值存为`code/PINN_10d/separate/u_pred.txt`文件。

 

## 3.3、DeepONet求解二维泊松方程

运行`code/DeepONet_2d/predict.py`，默认预测竞赛给定的测试集并画图，同时将预测值存为`code/DeepONet_2d/u_pred.txt`文件。

如果用文件的形式给定预测点，请将预测点文件放在`code/DeepONet_2d/data`文件夹中并命名为`points.txt`，预测点文件应该是`(N,2)`维数组，其中`N`是预测点的个数。并将`code/DeepONet_2d/predict.py`中的参数`is_file`设置为`True`，然后运行文件即可得到指定点处的值，同时将预测值存为`code/DeepONet_2d/u_pred.txt`文件。