"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
from torch.distributions import Bernoulli

def get_reward(action,target,target1) :
    if action == target :
        accuracy = 1
        if action == 1 : 
            reward =23996 / 5356
            tp=1
            tn=0
            fp=0
            fn=0
        elif action == 0 :
            reward =23996 / 18640
            tp=0
            tn=1
            fp=0
            fn=0
    else :
        accuracy = 0
            
            
            
        if action ==1:
            reward=-0.15
            tp=0
            tn=0
            fp=1
            fn=0
            
        elif action ==0:
            reward = target1 +0.1
            
            tp=0
            tn=0
            fp=0
            fn=1
    return reward,accuracy,tp,tn,fp,fn


def get_reward1(action,target) :
    if action == target :
        accuracy = 1
        if action == 1 : 
            reward = 23996 / 5356
            tp=1
            tn=0
            fp=0
            fn=0
        elif action == 0 :
            reward = 23996 / 18640
            tp=0
            tn=1
            fp=0
            fn=0
    else :
        accuracy = 0
        if action == 1 :
            reward = 0
            tp=0
            tn=0
            fp=1
            fn=0
        elif action == 0 :
            reward = 0
            tp=0
            tn=0
            fp=0
            fn=1
    
    return reward,accuracy,tp,tn,fp,fn


def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            rewards = 0
            accs=0
            for i_num in range(num_batch):
                #batch_size = inputs.size(0)
                #total += batch_size
                
                state = loader[i_num][1:24]
                
                inputs = Variable(torch.FloatTensor(state))
                targets = loader[i_num][24]
                target1=-1*loader[i_num][0]
                #if use_cuda:
                #    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                m = Bernoulli(outputs)
                
                action = m.sample()
                action = action.data.numpy().astype(int)[0]
                #action = torch.round(outputs).detach().numpy()
                #action = action.astype('int')[0]
                reward,accuracy,tp,tn,fp,fn=get_reward(action,targets,target1)
                
                if outputs==1.0:
                    reward=0
                elif outputs==0.0:
                    reward=0
                else:
                    reward = -m.log_prob(action) * reward
                
                rewards+=reward
                #total_loss += loss.item()*batch_size
                #_, predicted = torch.max(outputs.data, 1)
                accs +=accuracy
                #correct += predicted.eq(targets).sum().item()
                
                

        elif isinstance(criterion, nn.MSELoss):
            rewards = 0
            accs=0
            for i_num in range(num_batch):
                #batch_size = inputs.size(0)
                #total += batch_size
                
                state = loader[i_num][1:24]
                
                inputs = Variable(torch.FloatTensor(state))
                targets = loader[i_num][24]
                #if use_cuda:
                #    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                action = torch.round(outputs).detach().numpy()
                action = action.astype('int')[0]
                reward,accuracy,tp,tn,fp,fn=get_reward1(action,targets)
                
                m = Bernoulli(outputs)
                #print(outputs)
                
                if outputs==1.0:
                    reward=0
                elif outputs==0.0:
                    reward=0
                else:
                    reward=(targets*torch.log(outputs)+(1-targets)*torch.log(1-outputs)) 
                
                rewards+=-reward
                #total_loss += loss.item()*batch_size
                #_, predicted = torch.max(outputs.data, 1)
                accs +=accuracy
                #correct += predicted.eq(targets).sum().item()
#loss = weights[1] * (target * torch.log(output)) + \
 #              weights[0] * ((1 - target) * torch.log(1 - output))
    return rewards/num_batch, 100.*accs/num_batch
