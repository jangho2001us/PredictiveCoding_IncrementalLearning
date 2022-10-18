import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

import torch

import utils

torch.set_num_threads(2)

tstart = time.time()

# Arguments
parser = argparse.ArgumentParser(description='Continual learning (hat) with predictive coding')
parser.add_argument('--root_save', type=str,
                    default='./checkpoint')
parser.add_argument('--seed', type=int, default=123, help='(default=%(default)d)')
parser.add_argument('--experiment', default='mnist2', type=str,
                    choices=['mnist2', 'mnist5', 'pmnist', 'cifar', 'cifar5', 'split-cifar10', 'fmnist2'],
                    help='(default=%(default)s)')
parser.add_argument('--approach', default='ewc', type=str,
                    choices=['random', 'sgd', 'sgd-frozen', 'lwf', 'lfl', 'ewc', 'imm-mean', 'imm-mode', 'sgd-restart'],
                    help='(default=%(default)s)')
parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--nepochs', default=1, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')

# PC options
parser.add_argument('--error-type', type=str, default='FixedPred',
                    choices=['Strict', 'FixedPred', 'Exact', 'StochasticFixedPred'])
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--iter', type=int, default=20)

parser.add_argument('--save_freq', type=int, default='10')

args = parser.parse_args()

# define save name
header = os.path.join(args.root_save, datetime.now().strftime('%y%m%d_') + "pc")
args.output += '_' + args.experiment
args.output += '_' + args.approach
args.output += '_ep' + str(args.nepochs)
args.output += '_lr' + str(args.lr)
args.output += '_type' + str(args.error_type)
args.output += '_eta' + str(args.eta)
args.output += '_iter' + str(args.iter)
args.output += '_seed' + str(args.seed)
args.output = header + args.output
if not os.path.exists(args.output):
    os.makedirs(args.output)

print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print(args.output)
print('=' * 100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]');
    sys.exit()

# Args -- Experiment
if args.experiment == 'mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment == 'mnist5':
    from dataloaders import mnist5 as dataloader
elif args.experiment == 'pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment == 'cifar':
    from dataloaders import cifar as dataloader
    # from dataloaders import cifar_pc0 as dataloader
elif args.experiment == 'cifar5':
    from dataloaders import cifar5 as dataloader
elif args.experiment == 'split-cifar10':
    from dataloaders import split_cifar10 as dataloader
elif args.experiment == 'fmnist2':
    from dataloaders import fmnist2 as dataloader

# Args -- Approach
if args.approach == 'random':
    from approaches import random as approach
elif args.approach == 'sgd':
    if args.experiment == 'mnist2':
        from approaches_pc import sgd_mnist2 as approach
    elif args.experiment == 'mnist5':
        from approaches_pc import sgd_mnist5 as approach
    elif args.experiment == 'cifar':
        from approaches_pc import sgd_cifar as approach
    elif args.experiment == 'cifar5':
        from approaches_pc import sgd_cifar5 as approach
    elif args.experiment == 'split-cifar10':
        from approaches_pc import sgd_split_cifar10 as approach
    elif args.experiment == 'fmnist2':
        from approaches_pc import sgd_mnist2 as approach

elif args.approach == 'sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach == 'sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach == 'lwf':
    from approaches import lwf as approach
elif args.approach == 'lfl':
    from approaches import lfl as approach

elif args.approach == 'ewc':
    # from approaches import ewc as approach
    if args.experiment == 'mnist2':
        from approaches_pc import ewc_mnist2 as approach
    elif args.experiment == 'fmnist2':
        from approaches_pc import sgd_mnist2 as approach

elif args.approach == 'imm-mean':
    if args.experiment == 'mnist2':
        from approaches_pc import imm_mean_mnist2 as approach
elif args.approach == 'imm-mode':
    if args.experiment == 'mnist2':
        from approaches_pc import imm_mode_mnist2 as approach
elif args.approach == 'progressive':
    from approaches import progressive as approach
elif args.approach == 'pathnet':
    from approaches import pathnet as approach
elif args.approach == 'hat-test':
    from approaches import hat_test as approach
elif args.approach == 'hat':
    from approaches import hat as approach
elif args.approach == 'joint':
    from approaches import joint as approach

# Args -- Network
if args.experiment in ['mnist2', 'mnist5', 'pmnist', 'fmnist2']:
    if args.approach == 'hat' or args.approach == 'hat-test':
        from networks import mlp_hat as network
    else:
        from networks_pc import mlp as network

elif args.experiment == 'cifar5' or args.experiment == 'split-cifar10':
    from networks_pc import alexnet_cifar5 as network

else:
    if args.approach == 'lfl':
        from networks import alexnet_lfl as network
    elif args.approach == 'hat':
        from networks import alexnet_hat as network
    elif args.approach == 'progressive':
        from networks import alexnet_progressive as network
    elif args.approach == 'pathnet':
        from networks import alexnet_pathnet as network
    elif args.approach == 'hat-test':
        from networks import alexnet_hat_test as network
    else:
        # from networks import alexnet as network
        from networks_pc import alexnet as network

########################################################################################################################

# Load
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed)
print('Input size =', inputsize, '\nTask info =', taskcla)

# make task info
taskinfo = []
for item in taskcla:
    taskinfo.append(item[1])
print('TaskInfo: ', taskinfo)

# Inits
print('Inits...')
net = network.Net.cuda()
utils.print_model_report(net)

appr = approach.Appr(net, taskinfo, nepochs=args.nepochs, lr=args.lr,
                     error_type=args.error_type, eta=args.eta, iter=args.iter,
                     args=args)
print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t == 0:
            xtrain = data[t]['train']['x']
            ytrain = data[t]['train']['y']
            xvalid = data[t]['valid']['x']
            yvalid = data[t]['valid']['y']
            task_t = t * torch.ones(xtrain.size(0)).int()
            task_v = t * torch.ones(xvalid.size(0)).int()
            task = [task_t, task_v]
        else:
            xtrain = torch.cat((xtrain, data[t]['train']['x']))
            ytrain = torch.cat((ytrain, data[t]['train']['y']))
            xvalid = torch.cat((xvalid, data[t]['valid']['x']))
            yvalid = torch.cat((yvalid, data[t]['valid']['y']))
            task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int()))
            task = [task_t, task_v]
    else:
        # Get data
        xtrain = data[t]['train']['x'].cuda()
        ytrain = data[t]['train']['y'].cuda()
        xvalid = data[t]['valid']['x'].cuda()
        yvalid = data[t]['valid']['y'].cuda()
        task = t

    # Train
    appr.train(task, xtrain, ytrain, xvalid, yvalid)
    print('-' * 100)

    # Test
    for u in range(t + 1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'
              .format(u, data[u]['name'], test_loss, 100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    # Save
    print('Save at ' + args.output)
    np.savetxt(os.path.join(args.output, 'results.txt'), acc, '%.4f')

# Done
print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        # save task names
        from copy import deepcopy

        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t, ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t] = deepcopy(acc[t, :])
            appr.logs['test_loss'][t] = deepcopy(lss[t, :])
        # pickle
        import gzip
        import pickle

        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
