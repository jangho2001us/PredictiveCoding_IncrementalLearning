import torch
import torch.nn as nn

# import utils_pc

ncha = 1
size = 28

Net = nn.Sequential(
    # fc1
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(ncha * size * size, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # fc2
    nn.Sequential(
        nn.Linear(800, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # fc3
    nn.Sequential(
        nn.Linear(800, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),

    # last: task 0 (0-4) & task 1 (5-9)
    nn.Sequential(
        nn.Linear(800, 10)
    ),
)


Net_mnist2 = nn.Sequential(
    # fc1
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(ncha * size * size, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # fc2
    nn.Sequential(
        nn.Linear(800, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # fc3
    nn.Sequential(
        nn.Linear(800, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # last: task 0 (0-4) & task 1 (5-9)
    nn.Sequential(
        nn.Linear(800, 10)
    ),
)

Net_pmnist = nn.Sequential(
    # fc1
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(ncha * size * size, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # fc2
    nn.Sequential(
        nn.Linear(800, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # fc3
    nn.Sequential(
        nn.Linear(800, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # task0
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task1
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task2
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task3
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task4
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task5
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task6
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task7
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task8
    nn.Sequential(
        nn.Linear(800, 10)
    ),
    # task9
    nn.Sequential(
        nn.Linear(800, 10)
    ),


)