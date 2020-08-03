import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduction="batchmean")
    # outputs = F.log_softmax(inputs, dim=1)
    # print(outputs)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    # loss = loss.sum()/loss.shape[0] # batch average loss
    return loss

def L1_loss(inputs, labels):
    criterion = nn.L1Loss(reduction='mean')
    loss = criterion(inputs, labels.float())

    return loss

