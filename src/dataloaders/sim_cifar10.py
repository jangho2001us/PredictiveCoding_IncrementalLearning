import os, sys
import numpy as np
import torch
import utils
from torchvision import datasets, transforms
from sklearn.utils import shuffle


def get(seed=0, pc_valid=0.10):
    '''
    make CIFAR-10 to specific order
    task1: airplane, car
    task2: bird, cat
    task3: deer, dog
    task4: frog, horse
    task5: ship, truck
    '''
    data = {}
    taskcla = []
    size = [3, 32, 32]

    data_dir = os.path.join('../dat/sim_cifar10_seed{}'.format(seed))
    if not os.path.isdir(data_dir):
        print('create data dir: ', data_dir)
        os.makedirs(data_dir)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # CIFAR10
    dat = {}
    dat['train'] = datasets.CIFAR10('../dat',
                                    train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]))
    dat['test'] = datasets.CIFAR10('../dat',
                                   train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)]))

    for n in range(5):
        data[n] = {}
        data[n]['name'] = 'sim-cifar10'
        data[n]['ncla'] = 2
        data[n]['train'] = {'x': [], 'y': []}
        data[n]['test'] = {'x': [], 'y': []}
    for s in ['train', 'test']:
        loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)

        for image, target in loader:

            n = target.numpy()[0]

            # make sim-cifar10

            # task1
            if n == 1:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(0)
            elif n == 3:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(1)

            # task2
            elif n == 7:
                data[1][s]['x'].append(image)
                data[1][s]['y'].append(0)
            elif n == 9:
                data[1][s]['x'].append(image)
                data[1][s]['y'].append(1)

            # task3
            elif n == 5:
                data[2][s]['x'].append(image)
                data[2][s]['y'].append(0)
            elif n == 4:
                data[2][s]['x'].append(image)
                data[2][s]['y'].append(1)

            # task4
            elif n == 0:
                data[3][s]['x'].append(image)
                data[3][s]['y'].append(0)
            elif n == 2:
                data[3][s]['x'].append(image)
                data[3][s]['y'].append(1)

            # task5
            elif n == 6:
                data[4][s]['x'].append(image)
                data[4][s]['y'].append(0)
            elif n == 8:
                data[4][s]['x'].append(image)
                data[4][s]['y'].append(1)


    # "Unify" and save
    for t in data.keys():
        for s in ['train', 'test']:
            data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
            data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
            torch.save(data[t][s]['x'],
                       os.path.join(os.path.expanduser(data_dir), 'data' + str(t) + s + 'x.bin'))
            torch.save(data[t][s]['y'],
                       os.path.join(os.path.expanduser(data_dir), 'data' + str(t) + s + 'y.bin'))

    # Validation (new)
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
