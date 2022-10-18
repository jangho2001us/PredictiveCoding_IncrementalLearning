import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms

########################################################################################################################

def get(seed=0, fixed_order=False, pc_valid=0):
    data = {}
    taskcla = []
    size = [1, 28, 28]

    # MNIST
    mean = (0.1307,)
    std = (0.3081,)
    dat = {}
    dat['train'] = datasets.FashionMNIST('../dat/',
                                  train=True, download=True,
                                  transform=transforms.Compose(
                                      [transforms.ToTensor(),
                                       transforms.Normalize(mean, std)]))
    dat['test'] = datasets.FashionMNIST('../dat/',
                                 train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize(mean, std)]))

    for n in range(5):
        data[n] = {}
        data[n]['name'] = 'fmnist5'
        data[n]['ncla'] = 2
        data[n]['train'] = {'x': [], 'y': []}
        data[n]['test'] = {'x': [], 'y': []}

    for s in ['train', 'test']:
        loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        for image, target in loader:
            n = target.numpy()[0]
            nn = n // 2  # 0, 1, 2, 3, 4
            data[nn][s]['x'].append(image)
            data[nn][s]['y'].append(n % 2)  # 0, 1


    # "Unify" and save
    for n in [0, 1, 2, 3, 4]:
        for s in ['train', 'test']:
            data[n][s]['x'] = torch.stack(data[n][s]['x']).view(-1, size[0], size[1], size[2])
            data[n][s]['y'] = torch.LongTensor(np.array(data[n][s]['y'], dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size

########################################################################################################################
