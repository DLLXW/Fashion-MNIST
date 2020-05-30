from __future__ import print_function   # 从future版本导入print函数功能
import argparse                         # 加载处理命令行参数的库
import torch                            # 引入相关的包
import torch.nn.functional as F         # 引用神经网络常用函数包，不具有可学习的参数
import torch.optim as optim
from torchvision import datasets, transforms  # 加载pytorch官方提供的dataset
from tensorboardX import SummaryWriter
import os
from net import MLP,CNN#导入我们在net.py里面定义的网络
def main():
    # parser是训练和测试的一些参数设置，如果default里面有数值，则默认用它，
    # 要修改可以修改default，也可以在命令行输入
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', default='CNN',#这里选择你要训练的模型
                        help='CNN or MLP')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--save_dir', default='output/',#模型保存路径
                        help='dir saved models')
    args = parser.parse_args()
    #torch.cuda.is_available()会判断电脑是否有可用的GPU,没有则用cpu训练
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionmnist_data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionmnist_data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    writer=SummaryWriter()#用于记录训练和测试的信息:loss,acc等
    if args.model=='CNN':
        model = CNN().to(device)#CNN() or MLP
    if args.model=='MLP':
        model = MLP().to(device)#CNN() or MLP
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)   #optimizer存储了所有parameters的引用，每个parameter都包含gradient
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 24], gamma=0.1)   #学习率按区间更新
    model.train()
    log_loss=0
    log_acc=0
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)  # negative log likelihood loss(nll_loss), sum up batch cross entropy
            loss.backward()
            optimizer.step()  # 根据parameter的梯度更新parameter的值
            # 这里设置每args.log_interval个间隔打印一次训练信息，同时进行一次验证，并且将验证(测试)的准确率存入writer
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                #下面是模型验证过程
                model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():  # 无需计算梯度
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                test_loss /= len(test_loader.dataset)
                writer.add_scalars('loss', {'train_loss':loss,'val_loss':test_loss},global_step=log_acc)
                writer.add_scalar('val_accuracy', correct / len(test_loader.dataset), global_step=log_acc)
                log_acc += 1
                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))
                model.train()
    if (args.save_model):#保存训练好的模型
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir,args.model+".pt"))
    writer.add_graph(model, (data,))# 将模型结构保存成图，跟踪数据流动
    writer.close()
if __name__ == '__main__':
    main()