import torch
from net import MLP,CNN #
from torchvision import datasets, transforms
from sklearn.metrics import multilabel_confusion_matrix
#
test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionmnist_data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)
model=MLP()
device=torch.device('cpu')
model=model.to(device)
model.load_state_dict(torch.load('output/MLP.pt'))
model.eval()
pres=[]
labels=[]
i=0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    pres.append(pred[0][0].item())
    labels.append(target[0].item())
mcm = multilabel_confusion_matrix(labels, pres)#mcm
print(mcm)
