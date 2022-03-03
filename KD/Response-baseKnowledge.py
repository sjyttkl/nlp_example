# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     example.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2022/3/3 21:24
   Description :  https://nlplearning.blog.csdn.net/article/details/123261962
==================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
from matplotlib import pyplot as plt

import  argparse
torch.manual_seed(0)
# torch.cuda.manual_seed(0)


torch.manual_seed(34)
output = torch.randn(2, 3)
print(output)

print(F.softmax(output, dim=1))

print(torch.log(F.softmax(output, dim=1)))
print(F.log_softmax(output, dim=1))

# negative log likelihood
# 需要输入了log_sofmax ,target
print(F.nll_loss(torch.tensor([[-1.2, -2, -3]]), torch.tensor([0])))

# 通常 结合log_softmax 和 nll_loss 一起使用 ,这里是损失函数。交叉熵
output = torch.tensor([[1.2, 2, 3]])
target = torch.tensor([0])
log_sm_output = F.log_softmax(output, dim=1)
print(F.nll_loss(log_sm_output, target))
# 交叉熵和nll_loss 是一样的,只是 cross_entropy 里需要输入 output 没有经过激活函数
print(F.cross_entropy(output, target))


def sofmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)


output = np.array([0.1, 1.6, 3.6])
print(sofmax(output))


# t是温度的意思，其实就是 平滑作用
def softmax_t(x, t):
    x_exp = np.exp(x / t)
    return x_exp / np.sum(x_exp)


output = np.array([0.1, 1.6, 3.6])
print(softmax_t(output, 10))


# 设置默认参数
def args_parser():
    parser = argparse.ArgumentParser()      # 参数解析

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=50,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,      # 选择比例
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,     # 本地迭代次数
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,     # 本地的batch size
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,       # 学习率
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,     # SGD的动力参数
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')       # 日志显示，输出进度条记录
    parser.add_argument('--seed', type=int, default=1, help='random seed')      # 随机数种子
    args = parser.parse_args()
    return args




# 定义教师网络
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x) #没有经过softmax，
        return output


# 学生网络
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.relu(self.fc3(x))
        return output


# 训练教师网络一个epoch
def train_teacher(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # data的大小为64,1,28,28 ==> 即 batchsize = 64
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()  # 反向传播
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


# 测试集合测试一个epoch
def test_teacher(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


# 训练教师网络的主函数
def teacher_main():
    epochs = args.epochs
    batch_size = 512
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/songdongdong/workSpace/datas/pytorch', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/songdongdong/workSpace/datas/pytorch', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    model = TeacherNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    teacher_history = []

    for epoch in range(1, epochs + 1):
        train_teacher(model, device, train_loader, optimizer, epoch)  # 训练一个epoch
        loss, acc = test_teacher(model, device, test_loader)  # 测试集测试一个eopch

        teacher_history.append((loss, acc))  # 添加测试集的结果

    torch.save(model.state_dict(), "teacher.pt")
    return model, teacher_history


def plot_softmax():
    test_loader_bs1 = torch.utils.data.DataLoader(
        datasets.MNIST('../../data/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)
    teacher_model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader_bs1))
        data, target = data.to('cuda'), target.to('cuda')
        output = teacher_model(data)

    test_x = data.cpu().numpy()
    y_out = output.cpu().numpy()
    y_out = y_out[0, ::]
    print('Output (NO softmax):', y_out)

    plt.subplot(3, 1, 1)
    plt.imshow(test_x[0, 0, ::])

    plt.subplot(3, 1, 2)
    plt.bar(list(range(10)), softmax_t(y_out, 1), width=0.3)

    plt.subplot(3, 1, 3)
    plt.bar(list(range(10)), softmax_t(y_out, 10), width=0.3)
    plt.show()
    plt.close()


def softmax_t(x, t):
    x_exp = np.exp(x / t)
    return x_exp / np.sum(x_exp)


# 蒸馏网络的总Loss
def distillation(y, labels, teacher_scores, temp, alpha):
    """

    :param y: 学生的输出-没有经过softmax
    :param labels:  真是标签
    :param teacher_scores: 老师的输出（没有softamx）
    :param temp:  温度
    :param alpha:权重
    :return:  kl散度【（学生的输出，进行平滑）（老师的输出，进行平滑t）】 + 学生硬输出
    """
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)  # 蒸馏+交叉熵


# 训练蒸馏学生网络一个epoch
def train_student_kd(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        teacher_output = teacher_model(data)
        teacher_output = teacher_output.detach()  # 切断老师网络的反向传播，感谢B站“淡淡的落”的提醒
        loss = distillation(output, target, teacher_output, temp=5.0, alpha=0.7)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


# 测试蒸馏学生网络一个epoch
def test_student_kd(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


# 学生蒸馏网络的主函数
def student_kd_main():
    epochs = args.epochs
    batch_size = 512
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/songdongdong/workSpace/datas/pytorch/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/songdongdong/workSpace/datas/pytorch', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    model = StudentNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    student_history = []
    for epoch in range(1, epochs + 1):
        train_student_kd(model, device, train_loader, optimizer, epoch)  # 训练一个epoch
        loss, acc = test_student_kd(model, device, test_loader)  # 测试集测试一个epoch
        student_history.append((loss, acc))

    torch.save(model.state_dict(), "student_kd.pt")
    return model, student_history


# 学生网络训练一个epoch
def train_student(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


# 学生网络测试一个epoch
def test_student(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


# 学生网络主函数
def student_main():
    epochs = args.epochs
    batch_size = 512
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/songdongdong/workSpace/datas/pytorch', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/songdongdong/workSpace/datas/pytorch', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    model = StudentNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    student_history = []

    for epoch in range(1, epochs + 1):
        train_student(model, device, train_loader, optimizer, epoch)
        loss, acc = test_student(model, device, test_loader)
        student_history.append((loss, acc))

    torch.save(model.state_dict(), "student.pt")
    return model, student_history

args = args_parser()
# 训练教师网络
teacher_model, teacher_history = teacher_main()

# plot_softmax()  # 绘制蒸馏

student_kd_model, student_kd_history = student_kd_main()

student_simple_model, student_simple_history = student_main()

epochs = args.epochs
x = list(range(1, epochs + 1))
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, [teacher_history[i][1] for i in range(epochs)], label='teacher')
plt.plot(x, [student_kd_history[i][1] for i in range(epochs)], label='student with KD')
plt.plot(x, [student_simple_history[i][1] for i in range(epochs)], label='student without KD')

plt.title('Test accuracy')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x, [teacher_history[i][0] for i in range(epochs)], label='teacher')
plt.plot(x, [student_kd_history[i][0] for i in range(epochs)], label='student with KD')
plt.plot(x, [student_simple_history[i][0] for i in range(epochs)], label='student without KD')

plt.title('Test loss')
plt.legend()