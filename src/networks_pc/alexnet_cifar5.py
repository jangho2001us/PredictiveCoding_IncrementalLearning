import torch
import torch.nn as nn

ncha = 3
size = 32

s0 = size // 8
s1 = 14
s2 = 6
s3 = 2

Net = nn.Sequential(
    # conv1
    nn.Sequential(
        nn.Conv2d(ncha, 64, kernel_size=size // 8),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.MaxPool2d(kernel_size=2),
    ),
    # conv2
    nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=size // 10),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.MaxPool2d(kernel_size=2),
    ),
    # conv3
    nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
    ),

    # fc1
    nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
    ),

    # fc2
    nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
    ),

    # last: task 0 to 10
    nn.Sequential(
        nn.Linear(2048, 10)
    ),

)