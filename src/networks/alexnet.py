import sys
import torch

import utils


class Net(torch.nn.Module):

    def __init__(self, inputsize, taskcla):
        super(Net, self).__init__()

        ncha, size, _ = inputsize
        self.taskcla = taskcla

        # print('1: ', size // 8)
        self.conv1 = torch.nn.Conv2d(ncha, 64, kernel_size=size // 8)
        s = utils.compute_conv_output_size(size, size // 8)
        s = s // 2 # 14
        # print('s1: ', s)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=size // 10)
        s = utils.compute_conv_output_size(s, size // 10)
        s = s // 2 # 6
        # print('s2: ', s)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2)
        s = utils.compute_conv_output_size(s, 2)
        s = s // 2 # 2
        # print('s3: ', s)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(256 * s * s, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)
        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(2048, n))

        return

    def forward(self, x):
        h = self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h)))) # (64, 256, 2, 2)
        h = h.view(x.size(0), -1) # (64, 1024)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](h))
        return y


# inputsize = [3, 32, 32]
# taskcla = [(0, 2), (1, 20), (2, 2), (3, 20), (4, 2), (5, 20), (6, 20), (7, 2), (8, 2), (9, 20)]
# print(len(taskcla))
# model = Net(inputsize, taskcla)
#
# input = torch