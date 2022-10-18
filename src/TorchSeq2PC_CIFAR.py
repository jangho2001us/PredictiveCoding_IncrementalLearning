import numpy as np
import random

import torch

print('Running TorchSeq2PC_CIFAR.py')


def get_cifar_mask(taskinfo, task):
    # taskinfo = [2, 20, 2, 20, 2, 20, 20, 2, 2, 20]
    num_node = sum(taskinfo)

    mask = torch.zeros(num_node).long()

    if task == 0:
        start_idx = 0
    else:
        start_idx = sum(taskinfo[:task])
    end_idx = sum(taskinfo[:task + 1])
    # print(start_idx, end_idx)

    mask[start_idx:end_idx] += 1
    # print(mask)

    return mask


# mask = get_cifar_mask(task=9)


def get_cifar_target(taskinfo, task, Y):
    # taskinfo = [2, 20, 2, 20, 2, 20, 20, 2, 2, 20]

    Y = Y + sum(taskinfo[:task])

    return Y


# task = 1
# Y = torch.tensor([0, 1, 2])
# targets = get_cifar_target(task, Y)
# print(targets)

# NEWER

# Perform a forward pass on a Sequential model
# where X,Y are one batch of inputs,labels
# Returns activations for all layers (vhat), loss, and gradient of loss
# wrt last-layer activations (dLdy)
# vhat,Loss,dLdy=FwdPassPlus(model,LossFun,X,Y)
def FwdPassPlus(model, LossFun, X, Y):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    # Forward pass
    vhat = [None] * DepthPlusOne
    vhat[0] = X
    for layer in range(1, DepthPlusOne):
        f = model[layer - 1]
        vhat[layer] = f(vhat[layer - 1])
    Loss = LossFun(vhat[-1], Y)

    # Compute gradient of loss with respect to output
    dLdy = torch.autograd.grad(Loss, vhat[-1])[0]

    return vhat, Loss, dLdy


def FwdPassPlusCL_CIFAR(task, taskinfo, model, LossFun, X, Y):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    # Forward pass
    vhat = [None] * DepthPlusOne
    vhat[0] = X
    for layer in range(1, DepthPlusOne):
        f = model[layer - 1]
        vhat[layer] = f(vhat[layer - 1])

    # masking with task
    mask = get_cifar_mask(taskinfo, task).cuda()

    # calibrate target with task
    Y = get_cifar_target(taskinfo, task, Y).cuda()

    # apply mask
    vhat[-1] = mask * vhat[-1]
    # if task == 1:
    # print('=====================')
    # print(vhat[-1].shape, Y.shape)
    Loss = LossFun(vhat[-1], Y)

    # Compute gradient of loss with respect to output
    dLdy = torch.autograd.grad(Loss, vhat[-1])[0]

    return vhat, Loss, dLdy


def FwdPassPlusCLEWC(task, model, LossFun, X, Y):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    # Forward pass
    vhat = [None] * DepthPlusOne
    vhat[0] = X
    for layer in range(1, DepthPlusOne):
        f = model[layer - 1]
        vhat[layer] = f(vhat[layer - 1])

    # masking with task
    if task == 0:
        # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        submask1 = torch.ones(5).long().cuda()
        submask2 = torch.zeros(5).long().cuda()
        mask = torch.cat((submask1, submask2), dim=0)

    elif task == 1:
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        submask1 = torch.zeros(5).long().cuda()
        submask2 = torch.ones(5).long().cuda()
        mask = torch.cat((submask1, submask2), dim=0)

    # increase target indices
    if task == 1:
        Y = Y + 5

    # apply mask
    vhat[-1] = mask * vhat[-1]
    # if task == 1:
    # print('=====================')
    # print(vhat[-1].shape, Y.shape)
    Loss = LossFun(task, vhat[-1], Y)

    # Compute gradient of loss with respect to output
    dLdy = torch.autograd.grad(Loss, vhat[-1])[0]

    return vhat, Loss, dLdy


def FwdPassPlusIMMMEAN(task, model, LossFun, X, Y):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    # Forward pass
    vhat = [None] * DepthPlusOne
    vhat[0] = X
    for layer in range(1, DepthPlusOne):
        f = model[layer - 1]
        vhat[layer] = f(vhat[layer - 1])

    # masking with task
    if task == 0:
        # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        submask1 = torch.ones(5).long().cuda()
        submask2 = torch.zeros(5).long().cuda()
        mask = torch.cat((submask1, submask2), dim=0)

    elif task == 1:
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        submask1 = torch.zeros(5).long().cuda()
        submask2 = torch.ones(5).long().cuda()
        mask = torch.cat((submask1, submask2), dim=0)

    # increase target indices
    if task == 1:
        Y = Y + 5

    # apply mask
    vhat[-1] = mask * vhat[-1]
    # if task == 1:
    # print('=====================')
    # print(vhat[-1].shape, Y.shape)
    Loss = LossFun(vhat[-1], Y, task)

    # Compute gradient of loss with respect to output
    dLdy = torch.autograd.grad(Loss, vhat[-1])[0]

    return vhat, Loss, dLdy


def FwdPassPlusIMMMODE(task, model, LossFun, X, Y):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    # Forward pass
    vhat = [None] * DepthPlusOne
    vhat[0] = X
    for layer in range(1, DepthPlusOne):
        f = model[layer - 1]
        vhat[layer] = f(vhat[layer - 1])

    # masking with task
    if task == 0:
        # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        submask1 = torch.ones(5).long().cuda()
        submask2 = torch.zeros(5).long().cuda()
        mask = torch.cat((submask1, submask2), dim=0)

    elif task == 1:
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        submask1 = torch.zeros(5).long().cuda()
        submask2 = torch.ones(5).long().cuda()
        mask = torch.cat((submask1, submask2), dim=0)

    # increase target indices
    if task == 1:
        Y = Y + 5

    # apply mask
    vhat[-1] = mask * vhat[-1]
    # if task == 1:
    # print('=====================')
    # print(vhat[-1].shape, Y.shape)
    # print(task, vhat[-1], Y)
    Loss = LossFun(task, vhat[-1], Y)

    # Compute gradient of loss with respect to output
    dLdy = torch.autograd.grad(Loss, vhat[-1])[0]

    return vhat, Loss, dLdy


def FwdPassPlusCLLWF(task, model_old, model, LossFun, X, Y):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    if task > 0:
        # Forward pass for model_old
        vhat_old = [None] * DepthPlusOne
        vhat_old[0] = X
        for layer in range(1, DepthPlusOne):
            f = model_old[layer - 1]
            vhat_old[layer] = f(vhat_old[layer - 1])

    # Forward pass
    vhat = [None] * DepthPlusOne
    vhat[0] = X
    for layer in range(1, DepthPlusOne):
        f = model[layer - 1]
        vhat[layer] = f(vhat[layer - 1])

    # masking with task
    if task == 0:
        # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        submask1 = torch.ones(5).long().cuda()
        submask2 = torch.zeros(5).long().cuda()
        mask = torch.cat((submask1, submask2), dim=0)

    elif task == 1:
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        submask1 = torch.zeros(5).long().cuda()
        submask2 = torch.ones(5).long().cuda()
        mask = torch.cat((submask1, submask2), dim=0)

    # increase target indices
    if task == 1:
        Y = Y + 5

    # apply mask
    if task > 0:
        vhat_old[-1] = mask * vhat_old[-1]

    vhat[-1] = mask * vhat[-1]
    # print(vhat[-1].shape)

    outputs_old = [[vhat_old[-1][:5]], [vhat_old[-1][5:]]]
    outputs = [[vhat[-1][:5]], [vhat[-1][5:]]]

    if task == 0:
        # print('option a: ', task)
        # Loss = LossFun(task, None, vhat[-1], Y)
        Loss = LossFun(task, None, outputs, Y)
    elif task > 0:
        # Loss = LossFun(task, vhat_old[-1], vhat[-1], Y)
        Loss = LossFun(task, outputs_old, outputs, Y)

    # Compute gradient of loss with respect to output
    dLdy = torch.autograd.grad(Loss, vhat[-1])[0]

    return vhat, Loss, dLdy


# Compute prediction errors (epsilon) and beliefs (v)
# using predictive coding algorithm modified by
# the fixed prediction assumption
# see: Millidge, Tschantz, and Buckley. Predictive coding approximates backprop along arbitrary computation graphs.
# v,epsilon=FixedPredPCPredErrs(model,vhat,dLdy,eta=1,n=None)
def FixedPredPCPredErrs(model, vhat, dLdy, eta=1, n=None):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    if n == None:
        n = len(model)

    # Initialize epsilons
    epsilon = [None] * DepthPlusOne
    epsilon[-1] = dLdy

    # Initialize v to a copy of vhat with no gradients needed
    # (can this be moved up to the loop above?)
    v = [None] * DepthPlusOne
    for layer in range(DepthPlusOne):
        v[layer] = vhat[layer].clone().detach()

    # Iterative updates of v and epsilon using stored values of vhat
    for i in range(n):
        for layer in reversed(range(DepthPlusOne - 1)):  # range(DepthPlusOne-2,-1,-1):
            epsilon[layer] = vhat[layer] - v[layer]
            _, epsdfdv = torch.autograd.functional.vjp(model[layer], vhat[layer], epsilon[layer + 1])
            dv = epsilon[layer] - epsdfdv
            v[layer] = v[layer] + eta * dv

        # This helps free up memory
        with torch.no_grad():
            for layer in range(1, DepthPlusOne - 1):
                v[layer] = v[layer].clone()
                epsilon[layer] = epsilon[layer].clone()
    return v, epsilon


def StochasticFixedPredPCPredErrs(model, vhat, dLdy, eta=1, n=None):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    if n == None:
        n = len(model)

    # Initialize epsilons
    epsilon = [None] * DepthPlusOne
    epsilon[-1] = dLdy

    # Initialize v to a copy of vhat with no gradients needed
    # (can this be moved up to the loop above?)
    v = [None] * DepthPlusOne
    for layer in range(DepthPlusOne):
        v[layer] = vhat[layer].clone().detach()

    # Iterative updates of v and epsilon using stored values of vhat
    for i in range(n):
        if i == 0:
            for layer in reversed(range(DepthPlusOne - 1)):  # range(DepthPlusOne-2,-1,-1):
                # for layer in indices:
                epsilon[layer] = vhat[layer] - v[layer]
                _, epsdfdv = torch.autograd.functional.vjp(model[layer], vhat[layer], epsilon[layer + 1])
                dv = epsilon[layer] - epsdfdv
                v[layer] = v[layer] + eta * dv

        else:
            # shuffle layer index
            indices = np.arange(DepthPlusOne - 1)
            indices = indices.tolist()
            random.shuffle(indices)

            for layer in indices:
                epsilon[layer] = vhat[layer] - v[layer]
                _, epsdfdv = torch.autograd.functional.vjp(model[layer], vhat[layer], epsilon[layer + 1])
                dv = epsilon[layer] - epsdfdv
                v[layer] = v[layer] + eta * dv

        # This helps free up memory
        with torch.no_grad():
            for layer in range(1, DepthPlusOne - 1):
                v[layer] = v[layer].clone()
                epsilon[layer] = epsilon[layer].clone()
    return v, epsilon


# Compute prediction errors (epsilon) and beliefs (v)
# using a strict interpretation of predictive coding
# without the fixed prediction assumption.
# v,epsilon=StrictPCPredErrs(model,vinit,LossFun,Y,eta,n)
def StrictPCPredErrs(model, vinit, LossFun, Y, eta, n):
    with torch.no_grad():
        # Number of layers, counting the input as layer 0
        DepthPlusOne = len(model) + 1

        # Initialize epsilons
        epsilon = [None] * DepthPlusOne

        # Initialize v to a copy of vinit with no gradients needed
        # (can this be moved up to the loop above?)
        v = [None] * DepthPlusOne
        for layer in range(DepthPlusOne):
            v[layer] = vinit[layer].clone()

    # Iterative updates of v and epsilon
    for i in range(n):
        model.zero_grad()
        layer = DepthPlusOne - 1
        vtilde = model[layer - 1](v[layer - 1])
        Loss = LossFun(vtilde, Y)
        epsilon[layer] = torch.autograd.grad(Loss, vtilde, retain_graph=False)[0]  # -2 ~ DepthPlusOne-2
        for layer in reversed(range(1, DepthPlusOne - 1)):
            epsilon[layer] = v[layer] - model[layer - 1](v[layer - 1])
            _, epsdfdv = torch.autograd.functional.vjp(model[layer], v[layer], epsilon[layer + 1])
            dv = -epsilon[layer] + epsdfdv
            v[layer] = v[layer] + eta * dv
        # This helps free up memory
        with torch.no_grad():
            for layer in range(1, DepthPlusOne - 1):
                v[layer] = v[layer].clone()
                epsilon[layer] = epsilon[layer].clone()

    return v, epsilon


# Compute exact prediction errors (epsilon) and beliefs (v)
# epsilon is defined as the gradient of the loss wrt to
# the activations and v=vhat-epsilon where vhat are the
# activations from a forward pass.
# v,epsilon=ExactPredErrs(model,LossFun,X,Y,vhat=None)
def ExactPredErrs(model, LossFun, X, Y, vhat=None):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    # Forward pass if it wasn't passed in
    if vhat == None:
        vhat = [None] * DepthPlusOne
        vhat[0] = X
        for layer in range(1, DepthPlusOne):
            f = model[layer - 1]
            vhat[layer] = f(vhat[layer - 1])

    Loss = LossFun(vhat[-1], Y)

    epsilon = [None] * DepthPlusOne
    v = [None] * DepthPlusOne

    for layer in range(1, DepthPlusOne):
        epsilon[layer] = torch.autograd.grad(Loss, vhat[layer], allow_unused=True, retain_graph=True)[0]
        v[layer] = vhat[layer] - epsilon[layer]

    return v, epsilon


# Set gradients of model params based on PC approximations
def SetPCGrads(model, epsilon, X, vhat=None):
    # Number of layers, counting the input as layer 0
    DepthPlusOne = len(model) + 1

    # Forward pass if it wasn't passed in
    if vhat == None:
        vhat = [None] * DepthPlusOne
        vhat[0] = X
        for layer in range(1, DepthPlusOne):
            f = model[layer - 1]
            vhat[layer] = f(vhat[layer - 1])

    # Compute new parameter values
    for layer in range(0, DepthPlusOne - 1):
        for p in model[layer].parameters():
            dtheta = torch.autograd.grad(vhat[layer + 1], p, grad_outputs=epsilon[layer + 1], allow_unused=True,
                                         retain_graph=True)[0]
            p.grad = dtheta


# Perform a whole PC inference step
# Returns activations (vhat), loss, gradient of the loss wrt output (dLdy),
# beliefs (v), and prediction errors (epsilon)
# vhat,Loss,dLdy,v,epsilon=PCInfer(model,LossFun,X,Y,ErrType="FixedPred",eta=.1,n=20,vinit=None)
def PCInfer(model, LossFun, X, Y, ErrType, eta=.1, n=20, vinit=None):
    # Fwd pass (plus return vhat and dLdy)
    vhat, Loss, dLdy = FwdPassPlus(model, LossFun, X, Y)

    # Get beliefs and prediction errors
    if ErrType == "FixedPred":
        v, epsilon = FixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    elif ErrType == "Strict":
        if vinit == None:
            vinit = vhat
        v, epsilon = StrictPCPredErrs(model, vhat, LossFun, Y, eta, n)
    elif ErrType == "Exact":
        v, epsilon = ExactPredErrs(model, LossFun, X, Y)
    elif ErrType == 'StochasticFixedPred':
        v, epsilon = StochasticFixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    else:
        raise ValueError('ErrType must be \"FixedPred\", \"Strict\", or \"Exact\"')

    # Set gradients in model
    SetPCGrads(model, epsilon, X, vhat)

    return vhat, Loss, dLdy, v, epsilon


def PCInferCL_CIFAR(task, taskinfo, model, LossFun, X, Y, ErrType, eta=.1, n=20, vinit=None):
    # Fwd pass (plus return vhat and dLdy)
    # vhat, Loss, dLdy = FwdPassPlus(model, LossFun, X, Y)
    vhat, Loss, dLdy = FwdPassPlusCL_CIFAR(task, taskinfo, model, LossFun, X, Y)

    # Get beliefs and prediction errors
    if ErrType == "FixedPred":
        v, epsilon = FixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    elif ErrType == "Strict":
        if vinit == None:
            vinit = vhat
        v, epsilon = StrictPCPredErrs(model, vhat, LossFun, Y, eta, n)
    elif ErrType == "Exact":
        v, epsilon = ExactPredErrs(model, LossFun, X, Y)
    elif ErrType == 'StochasticFixedPred':
        v, epsilon = StochasticFixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    else:
        raise ValueError('ErrType must be \"FixedPred\", \"Strict\", or \"Exact\"')

    # Set gradients in model
    SetPCGrads(model, epsilon, X, vhat)

    return vhat, Loss, dLdy, v, epsilon


def PCInferCLEWC(task, model, LossFun, X, Y, ErrType, eta=.1, n=20, vinit=None):
    # Fwd pass (plus return vhat and dLdy)
    # vhat, Loss, dLdy = FwdPassPlus(model, LossFun, X, Y)
    vhat, Loss, dLdy = FwdPassPlusCLEWC(task, model, LossFun, X, Y)

    # Get beliefs and prediction errors
    if ErrType == "FixedPred":
        v, epsilon = FixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    elif ErrType == "Strict":
        if vinit == None:
            vinit = vhat
        v, epsilon = StrictPCPredErrs(model, vhat, LossFun, Y, eta, n)
    elif ErrType == "Exact":
        v, epsilon = ExactPredErrs(model, LossFun, X, Y)
    elif ErrType == 'StochasticFixedPred':
        v, epsilon = StochasticFixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    else:
        raise ValueError('ErrType must be \"FixedPred\", \"Strict\", or \"Exact\"')

    # Set gradients in model
    SetPCGrads(model, epsilon, X, vhat)

    return vhat, Loss, dLdy, v, epsilon


def PCInferCLLWF(task, model_old, model, LossFun, X, Y, ErrType, eta=.1, n=20, vinit=None):
    # Fwd pass (plus return vhat and dLdy)
    # vhat, Loss, dLdy = FwdPassPlus(model, LossFun, X, Y)
    vhat, Loss, dLdy = FwdPassPlusCLLWF(task, model_old, model, LossFun, X, Y)

    # Get beliefs and prediction errors
    if ErrType == "FixedPred":
        v, epsilon = FixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    elif ErrType == "Strict":
        if vinit == None:
            vinit = vhat
        v, epsilon = StrictPCPredErrs(model, vhat, LossFun, Y, eta, n)
    elif ErrType == "Exact":
        v, epsilon = ExactPredErrs(model, LossFun, X, Y)
    elif ErrType == 'StochasticFixedPred':
        v, epsilon = StochasticFixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    else:
        raise ValueError('ErrType must be \"FixedPred\", \"Strict\", or \"Exact\"')

    # Set gradients in model
    SetPCGrads(model, epsilon, X, vhat)

    return vhat, Loss, dLdy, v, epsilon


def PCInferIMMMEAN(task, model, LossFun, X, Y, ErrType, eta=.1, n=20, vinit=None):
    # Fwd pass (plus return vhat and dLdy)
    # vhat, Loss, dLdy = FwdPassPlus(model, LossFun, X, Y)
    vhat, Loss, dLdy = FwdPassPlusIMMMEAN(task, model, LossFun, X, Y)

    # Get beliefs and prediction errors
    if ErrType == "FixedPred":
        v, epsilon = FixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    elif ErrType == "Strict":
        if vinit == None:
            vinit = vhat
        v, epsilon = StrictPCPredErrs(model, vhat, LossFun, Y, eta, n)
    elif ErrType == "Exact":
        v, epsilon = ExactPredErrs(model, LossFun, X, Y)
    elif ErrType == 'StochasticFixedPred':
        v, epsilon = StochasticFixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    else:
        raise ValueError('ErrType must be \"FixedPred\", \"Strict\", or \"Exact\"')

    # Set gradients in model
    SetPCGrads(model, epsilon, X, vhat)

    return vhat, Loss, dLdy, v, epsilon


def PCInferIMMMODE(task, model, LossFun, X, Y, ErrType, eta=.1, n=20, vinit=None):
    # Fwd pass (plus return vhat and dLdy)
    # vhat, Loss, dLdy = FwdPassPlus(model, LossFun, X, Y)
    vhat, Loss, dLdy = FwdPassPlusIMMMODE(task, model, LossFun, X, Y)

    # Get beliefs and prediction errors
    if ErrType == "FixedPred":
        v, epsilon = FixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    elif ErrType == "Strict":
        if vinit == None:
            vinit = vhat
        v, epsilon = StrictPCPredErrs(model, vhat, LossFun, Y, eta, n)
    elif ErrType == "Exact":
        v, epsilon = ExactPredErrs(model, LossFun, X, Y)
    elif ErrType == 'StochasticFixedPred':
        v, epsilon = StochasticFixedPredPCPredErrs(model, vhat, dLdy, eta, n)
    else:
        raise ValueError('ErrType must be \"FixedPred\", \"Strict\", or \"Exact\"')

    # Set gradients in model
    SetPCGrads(model, epsilon, X, vhat)

    return vhat, Loss, dLdy, v, epsilon
