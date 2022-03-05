import torch
from torchvision import datasets,transforms
import torchvision
from torch.autograd import  Variable
import numpy as np
import matplotlib.pyplot as plt

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

data_train=datasets.MNIST(root="./data",  transform=data_tf, train=True,
                          download=True
                          )
data_test=datasets.MNIST(root="./data", transform=data_tf, train=False)

data_loader_train=torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=64,
                                              shuffle=True)
data_loader_test=torch.utils.data.DataLoader(dataset=data_test,
                                             batch_size=64,
                                             shuffle=True)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(1024,10))
    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1,14*14*128)
        x = self.dense(x)
        return x

model = Model()
if torch.cuda.is_available():
    model.cuda()#将所有的模型参数移动到GPU上
cost = torch.nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters())

n_epochs = 5

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for data in data_loader_train:
        # print("train ing")
        X_train, y_train = data
        # 有GPU加下面这行，没有不用加
        X_train, y_train = X_train.cuda(), y_train.cuda()
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimzer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimzer.step()
        running_loss += loss.data[0]
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        # 有GPU加下面这行，没有不用加
        X_test, y_test = X_test.cuda(), y_test.cuda()
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                     100 * running_correct / len(
                                                                                         data_train),
                                                                                     100 * testing_correct / len(
                                                                                         data_test)))