#定义网络结构
import torch                            # 引入相关的包
import torch.nn as nn                   # 指定torch.nn别名nn
import torch.nn.functional as F         # 引用神经网络常用函数包，不具有可学习的参数
#这里定义的是多层感知机
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 2000) #784表示输入神经元数量,2000表示这一层输出神经元数量
        self.fc2 = nn.Linear(2000, 1000)#第二层
        self.fc3 = nn.Linear(1000, 500)#第三层
        self.fc4 = nn.Linear(500, 100)
        self.fc5 = nn.Linear(100, 10)#最后一层直接输出10个类别的概率

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)  #
#定义CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1, 28, 28)
            nn.Conv2d(
                in_channels=1, # 输入通道数，若图片为RGB则为3通道
                out_channels=32, # 输出通道数，即多少个卷积核一起卷积
                kernel_size=3, # 卷积核大小
                stride=1, # 卷积核移动步长
                padding=1, # 边缘增加的像素，使得得到的图片长宽没有变化
            ),# (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), # (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 池化 (32, 14, 14)
        )
        self.conv3 = nn.Sequential(# (32, 14, 14)
            nn.Conv2d(32, 64, 3, 1, 1),# (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),# (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),# (64, 7, 7)
        )
        self.out = nn.Sequential(
            nn.Dropout(p = 0.5), # 抑制过拟合
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # (batch_size, 64*7*7)
        x = self.out(x)
        return F.log_softmax(x, dim=1)  #output
