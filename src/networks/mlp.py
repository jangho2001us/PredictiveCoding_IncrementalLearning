import sys
import torch


# import utils

class Net(torch.nn.Module):

    def __init__(self, inputsize, taskcla):
        super(Net, self).__init__()
        ncha, size, _ = inputsize
        self.taskcla = taskcla

        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(ncha * size * size, 800)
        self.fc2 = torch.nn.Linear(800, 800)
        self.fc3 = torch.nn.Linear(800, 800)
        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(800, n))

        return

    def forward(self, x):
        h = x.view(x.size(0), -1)  # (bs, 784)
        h = self.drop(self.relu(self.fc1(h)))  # (bs, 800)
        h = self.drop(self.relu(self.fc2(h)))  # (bs, 800)
        h = self.drop(self.relu(self.fc3(h)))  # (bs, 800)

        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](h))
        return y
    
    def forward_fc3(self, x):
        h = x.view(x.size(0), -1)  # (bs, 784)
        h = self.drop(self.relu(self.fc1(h)))  # (bs, 800)
        h = self.drop(self.relu(self.fc2(h)))  # (bs, 800)
        h = self.drop(self.relu(self.fc3(h)))  # (bs, 800)

        return h

# test
# inputsize = [1, 28, 28]
# taskcla = [(0, 5), (1, 5)]
# model = Net(inputsize, taskcla)
# print(model)
# for name, param in model.named_parameters():
#     print(name)
# input = torch.randn(8, 1, 28, 28)
# out = model(input)
# print(out[0].shape, out[1].shape)
