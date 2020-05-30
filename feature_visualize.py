#特征图可视化脚本
import torch
import cv2
from PIL import Image
from net import MLP,CNN #
from torchvision import datasets, transforms
import os
#
f_dir='features/'#存储可视化的特征图的路径
if not os.path.exists(f_dir):
    os.makedirs(f_dir)
def plot_x(x,w_dir):
    x = torch.squeeze(x)
    x=x.permute(1, 2, 0)
    cv2.imwrite(os.path.join(f_dir, w_dir), 255 * x.mean(2).unsqueeze(2).cpu().detach().numpy())
def feature_visualization(net,x):
    w_dirs=['layer_'+str(i)+'.jpg' for i in range(1,5)]
    x=net.conv1(x)
    plot_x(x,w_dirs[0])
    x = net.conv2(x)
    plot_x(x, w_dirs[1])
    x = net.conv3(x)
    plot_x(x, w_dirs[2])
    x = net.conv4(x)
    plot_x(x, w_dirs[3])

model=CNN()#
device=torch.device('cpu')#用cpu进行推理
model=model.to(device)
model.load_state_dict(torch.load('output/CNN.pt'))##
model.eval()#告诉模型验证
print(model)#可以打印网络结构观察
#--------以上就是推理之前模型的导入--------
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
img=Image.open('test_image.jpg')#用于推理的图片
image=data_transforms(img)#预处理，转成tensor同时正则化
image=image.unsqueeze(0)#[1,28,28]->[1,1,28,28]
feature_visualization(model,image.to(device))#可视化


