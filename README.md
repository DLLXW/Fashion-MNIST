# Fashion-MNIST
这是一个入门机器学习/深度学习的小项目，包含随机森林，多层感知器，卷积神经网络，实现了网络训练，验证，推理，训练可视化，多分类混淆矩阵计算，特征图可视化等，适合新手入门！[知乎链接](https://zhuanlan.zhihu.com/p/144567683)
## 环境说明

- scikit-learn
- pytorch,torchvision
- opencv
- pillow
- tensorboardx

## 使用步骤

### 深度学习相关

首先下载该工程到本地

1.训练模型

​		运行deep_learning.py：将会自动下载fashion mnist数据，然后开始训练.可以选择用CNN或者MLP

2.训练过程可视化

​		上一步训练完成后会再当前目录下生成runs目录(存放了tensorboard记录的训练信息),以及output目录(存放了输出模型)
 在命令行执行

```
tensorboard --logdir=./runs
```

输出的网址用浏览器打开即可看到loss的可视化结果和网络结构的可视化结构

3.模型推理

​        运行infer.py

4.打印验证集的混淆矩阵

​	    运行mcm.py

### 机器学习相关

运行machine_learning.py：可以看到打印出的分类信息和多分类混淆矩阵(为了避免自己手动下载数据，建议先运行deep_learning.py后再运行运行machine_learning.py)

