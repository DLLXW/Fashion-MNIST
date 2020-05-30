#
import torch
import cv2
from PIL import Image
from net import MLP,CNN #
from torchvision import datasets, transforms
#
model=MLP()#这里可选CNN(),要看你前面训练的是哪个
device=torch.device('cpu')#用cpu进行推理
model=model.to(device)
model.load_state_dict(torch.load('output/MLP.pt'))
model.eval()#这一步很重要，这是告诉模型我们要验证，而不是训练
#--------以上就是推理之前模型的导入--------
print("-------加载模型成功----------")
class_dic={0:"T恤",1:"裤子",2:"套头衫",3:"连衣裙",4:"外套",5:"凉鞋",6:"衬衫",7:"运动鞋",8:"包",9:"靴子"}
data_transforms = transforms.Compose([
    #transforms.ToTensor() convert a PIL image to tensor (HWC) in range [0,255] to a
    #torch.Tensor(CHW)in the range [0.0,1.0]
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
img=Image.open('test_image.jpg')#用于推理的图片
image=data_transforms(img)#预处理，转成tensor同时正则化
image=image.unsqueeze(0)#[1,28,28]->[1,1,28,28]
output = model(image.to(device))
pred = output.argmax(dim=1, keepdim=True)#
cls=pred.item()#输出在0~10之间代表10个类别
print("分类结果:",class_dic[cls])
