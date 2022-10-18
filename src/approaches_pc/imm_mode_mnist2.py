import sys, time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

import utils

import TorchSeq2PC as T2PC


class Appr(object):
    """ Class implementing the Incremental Moment Matching (mode) approach described in https://arxiv.org/abs/1703.08475 """

    def __init__(self, model,
                 nepochs=100, sbatch=64,
                 lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1000,
                 lamb=0.01,
                 error_type='FixedPred', eta=0.1, iter=20,
                 args=None):

        self.args = args
        self.model = model
        self.model_old = None
        self.fisher = None

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.error_type = error_type
        self.eta = eta
        self.iter = iter

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.lamb = lamb  # Grid search = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]; best was 1
        # self.alpha=0.5     # We assume the same alpha for all tasks. Unrealistic to tune one alpha per task (num_task-1 alphas) when we have a lot of tasks.

        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to', params)
            self.lamb = float(params[0])

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, xtrain, ytrain)
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, xtrain, ytrain)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / xtrain.size(0),
                1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0), train_loss, 100 * train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)

            # save model
            if e % self.args.save_freq == 0:
                utils.save_checkpoint(self.args, {
                    'epoch': e + 1,
                    'arch': self.args.approach,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=True, filename='checkpoint_task{:02d}_{:04d}.pth.tar'.format(t, e))

            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        # save best model
        utils.save_checkpoint(self.args, {
            'epoch': e + 1,
            'arch': self.args.approach,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=True, filename='final_best_model.pth.tar')

        # Model update
        if t == 0:
            self.fisher = utils.fisher_matrix_diag_pc(t, xtrain, ytrain, self.model, self.criterion,
                                                   self.error_type, self.eta, self.iter)
        else:
            fisher_new = utils.fisher_matrix_diag_pc(t, xtrain, ytrain, self.model, self.criterion,
                                                     self.error_type, self.eta, self.iter)
            for (n, p), (_, p_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                p = fisher_new[n] * p + self.fisher[n] * p_old
                self.fisher[n] += fisher_new[n]
                p /= (self.fisher[n] == 0).float() + self.fisher[n]

        # Old model save
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)

        return

    def train_epoch(self, t, x, y):
        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)

            # Forward current model
            # outputs = self.model.forward(images)
            # output = outputs[t]
            # loss = self.criterion(t, output, targets)
            #
            # # Backward
            # self.optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            # self.optimizer.step()

            # Forward with PC
            vhat, Loss, dLdy, v, epsilon = T2PC.PCInferIMMMODE(t, self.model, self.criterion,
                                                          images, targets,
                                                          self.error_type, self.eta, self.iter)

            # Backward with PC
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            self.model.zero_grad()
            self.optimizer.zero_grad()

        return

    def eval(self, t, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)

            # increase target indices
            if t == 1:
                targets = targets + 5

            # Forward
            # outputs = self.model.forward(images)
            # output = outputs[t]
            # loss = self.criterion(t, output, targets)
            # _, pred = output.max(1)
            # hits = (pred == targets).float()

            # Forward with PC
            outputs = self.model(images)

            # masking with task
            if t == 0:
                # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                submask1 = torch.ones(5).long().cuda()
                submask2 = torch.zeros(5).long().cuda()
                mask = torch.cat((submask1, submask2), dim=0)
            elif t == 1:
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
                submask1 = torch.zeros(5).long().cuda()
                submask2 = torch.ones(5).long().cuda()
                mask = torch.cat((submask1, submask2), dim=0)

            outputs = mask * outputs

            loss = self.criterion(t, outputs, targets)

            _, pred = outputs.max(1)
            hits = (pred == targets).float()

            # Log
            # total_loss+=loss.data.cpu().numpy()[0]*len(b)
            # total_acc+=hits.sum().data.cpu().numpy()[0]
            total_loss += loss.item() * len(b)
            total_acc += hits.sum()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):

        # L2 multiplier loss
        loss_reg = 0

        if t > 0:
            for p, p_old in zip(self.model.parameters(), self.model_old.parameters()):
                loss_reg += (p - p_old).pow(2).sum() / 2

        # Cross entropy loss
        loss_ce = self.ce(output, targets)

        return loss_ce + self.lamb * loss_reg


class ApprPlot(object):
    """ Class implementing the Incremental Moment Matching (mode) approach described in https://arxiv.org/abs/1703.08475 """

    def __init__(self, model, taskinfo,
                 nepochs=100, sbatch=64,
                 lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1000,
                 lamb=0.01,
                 error_type='FixedPred', eta=0.1, iter=20,
                 args=None):

        self.args = args
        self.model = model
        self.model_old = None
        self.fisher = None
        self.taskinfo = taskinfo

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.error_type = error_type
        self.eta = eta
        self.iter = iter

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.lamb = lamb  # Grid search = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]; best was 1
        # self.alpha=0.5     # We assume the same alpha for all tasks. Unrealistic to tune one alpha per task (num_task-1 alphas) when we have a lot of tasks.

        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to', params)
            self.lamb = float(params[0])

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, sf):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, xtrain, ytrain)
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, xtrain, ytrain)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / xtrain.size(0),
                1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0), train_loss, 100 * train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)

            # save model
            if e % self.args.save_freq == 0:
                utils.save_checkpoint(self.args, {
                    'epoch': e + 1,
                    'arch': self.args.approach,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=True, filename='checkpoint_task{:02d}_{:04d}.pth.tar'.format(t, e))

            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        # save best model
        utils.save_checkpoint(self.args, {
            'epoch': e + 1,
            'arch': self.args.approach,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=True, filename='final_best_model.pth.tar')

        # Model update
        if t == 0:
            self.fisher = utils.fisher_matrix_diag_pc(t, xtrain, ytrain, self.model, self.criterion,
                                                   self.error_type, self.eta, self.iter)
        else:
            fisher_new = utils.fisher_matrix_diag_pc(t, xtrain, ytrain, self.model, self.criterion,
                                                     self.error_type, self.eta, self.iter)
            for (n, p), (_, p_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                p = fisher_new[n] * p + self.fisher[n] * p_old
                self.fisher[n] += fisher_new[n]
                p /= (self.fisher[n] == 0).float() + self.fisher[n]

        # Old model save
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)

        return

    def train_epoch(self, t, x, y):
        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)

            # Forward current model
            # outputs = self.model.forward(images)
            # output = outputs[t]
            # loss = self.criterion(t, output, targets)
            #
            # # Backward
            # self.optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            # self.optimizer.step()

            # Forward with PC
            vhat, Loss, dLdy, v, epsilon = T2PC.PCInferIMMMODE(t, self.taskinfo,  self.model, self.criterion,
                                                          images, targets,
                                                          self.error_type, self.eta, self.iter)

            # Backward with PC
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            self.model.zero_grad()
            self.optimizer.zero_grad()

        return

    def eval(self, t, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)

            # increase target indices
            if t == 1:
                targets = targets + 5

            # Forward
            # outputs = self.model.forward(images)
            # output = outputs[t]
            # loss = self.criterion(t, output, targets)
            # _, pred = output.max(1)
            # hits = (pred == targets).float()

            # Forward with PC
            outputs = self.model(images)

            # masking with task
            if t == 0:
                # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                submask1 = torch.ones(5).long().cuda()
                submask2 = torch.zeros(5).long().cuda()
                mask = torch.cat((submask1, submask2), dim=0)
            elif t == 1:
                # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
                submask1 = torch.zeros(5).long().cuda()
                submask2 = torch.ones(5).long().cuda()
                mask = torch.cat((submask1, submask2), dim=0)

            outputs = mask * outputs

            loss = self.criterion(t, outputs, targets)

            _, pred = outputs.max(1)
            hits = (pred == targets).float()

            # Log
            # total_loss+=loss.data.cpu().numpy()[0]*len(b)
            # total_acc+=hits.sum().data.cpu().numpy()[0]
            total_loss += loss.item() * len(b)
            total_acc += hits.sum()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):

        # L2 multiplier loss
        loss_reg = 0

        if t > 0:
            for p, p_old in zip(self.model.parameters(), self.model_old.parameters()):
                loss_reg += (p - p_old).pow(2).sum() / 2

        # Cross entropy loss
        loss_ce = self.ce(output, targets)

        return loss_ce + self.lamb * loss_reg
