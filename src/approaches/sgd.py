import sys, time
import numpy as np
import torch

import utils


class Appr(object):

    def __init__(self, model,
                 nepochs=100, sbatch=64,
                 lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000,
                 args=None):

        self.args = args
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

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
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms '
                  '| Train: loss={:.3f}, acc={:5.1f}% '
                  '|'.format(e + 1, 1000 * self.sbatch * (
                    clock1 - clock0) / xtrain.size(0),
                             1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0),
                             train_loss,
                             100 * train_acc),
                  end='')
            clock2 = time.time()

            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'
                  .format(valid_loss, 100 * valid_acc), end='')
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
            # (64, 1, 28, 28) ... mnist
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)
            # print(images.shape, targets.shape)

            # Forward
            outputs = self.model.forward(images)
            output = outputs[t]
            loss = self.criterion(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

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

            # Forward
            outputs = self.model.forward(images)
            output = outputs[t]  # (bs, 5)
            loss = self.criterion(output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            # total_loss+=loss.data.cpu().numpy()[0]*len(b)
            total_loss += loss.item() * len(b)  # jangho
            # total_acc+=hits.sum().data.cpu().numpy()[0]
            total_acc += hits.sum()  # jangho
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    #
    # def eval_fc3(self, t, x, y):
    #     '''
    #     return the ouptut of fc3 in mlp
    #     '''
    #     self.model.eval()
    #
    #     r = np.arange(x.size(0))
    #     r = torch.LongTensor(r).cuda()
    #     print('r: ', r)
    #     # Loop batches
    #     for i in range(0, len(r), self.sbatch):
    #         if i + self.sbatch <= len(r):
    #             b = r[i:i + self.sbatch]
    #         else:
    #             b = r[i:]
    #         images = torch.autograd.Variable(x[b], volatile=True)
    #         targets = torch.autograd.Variable(y[b], volatile=True)
    #
    #         # Forward until fc3
    #         feats = self.model.forward_fc3(images)
    #
    #     if t == 1:
    #         targets = targets + 5
    #     print(feats.shape, targets.shape)
    #
    #     return feats, targets

    def eval_fc3(self, x, y):
        '''
        return the output of fc3 in mlp
        '''
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        feats_all = []
        targets_all = []
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)

            # Forward until fc3
            feat = self.model.forward_fc3(images)

            feats_all.append(feat)
            targets_all.append(targets)

        feats = torch.stack(feats_all, dim=0)
        targets = torch.stack(targets_all, dim=0)

        feats = feats.view(-1, 800)
        targets = targets.reshape(-1)

        return feats, targets


class ApprPlot(object):

    def __init__(self, model,
                 nepochs=100, sbatch=64,
                 lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000,
                 args=None):

        self.args = args
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, sf):
        '''
        data: all data containing train, valid, test
        sf: save file
        '''

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
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms '
                  '| Train: loss={:.3f}, acc={:5.1f}% '
                  '|'.format(e + 1, 1000 * self.sbatch * (
                    clock1 - clock0) / xtrain.size(0),
                             1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0),
                             train_loss,
                             100 * train_acc),
                  end='')
            clock2 = time.time()

            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'
                  .format(valid_loss, 100 * valid_acc), end='')

            # calculate test accuracy for each task
            saveline = str(e) + ' '
            with torch.no_grad():
                for u in range(t + 1):
                    xtest = data[u]['test']['x'].cuda()
                    ytest = data[u]['test']['y'].cuda()
                    test_loss, test_acc = self.eval(u, xtest, ytest)
                    print('>>> Intermediate Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'
                          .format(u, data[u]['name'], test_loss, 100 * test_acc))

                    temp = str((100 * test_acc).cpu().numpy())
                    saveline += temp + '\t'

            sf.write(saveline)
            sf.write('\n')

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
                        # break
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
        #
        # # calculate validation accuracy
        # with torch.no_grad():
        #     for u in range(t + 1):
        #         xtest = data[u]['test']['x'].cuda()
        #         ytest = data[u]['test']['y'].cuda()
        #         test_loss, test_acc = self.eval(u, xtest, ytest)
        #         print('>>> Intermediate Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'
        #               .format(u, data[u]['name'], test_loss, 100 * test_acc))
            # acc[t, u] = test_acc
            # lss[t, u] = test_loss

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
            # (64, 1, 28, 28) ... mnist
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)
            # print(images.shape, targets.shape)

            # Forward
            outputs = self.model.forward(images)
            output = outputs[t]
            loss = self.criterion(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

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

            # Forward
            outputs = self.model.forward(images)
            output = outputs[t]  # (bs, 5)
            loss = self.criterion(output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            # total_loss+=loss.data.cpu().numpy()[0]*len(b)
            total_loss += loss.item() * len(b)  # jangho
            # total_acc+=hits.sum().data.cpu().numpy()[0]
            total_acc += hits.sum()  # jangho
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num
