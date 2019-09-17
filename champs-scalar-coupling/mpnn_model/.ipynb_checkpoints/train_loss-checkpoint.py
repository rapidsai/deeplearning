import numpy as np 
import pandas as pd 

import torch 
from torch import nn 
import torch.nn.functional as F

#############################################################################################################
#                                                                                                           #
#                                                 Loss functions                                             #
#                                                                                                           #
#############################################################################################################
# lmae for single model: 1-type prediction 
def lmae_criterion(predict, coupling_value, coupling_rank, coupling_contribution, coupling_type,):
    '''
    lmae between regression predictions and true scalar coupling constant 
    '''
    coupling_preds, contribution_preds, type_preds = predict 
    predict = coupling_preds.view(-1)
    truth   = coupling_value.view(-1)
    assert(predict.shape==truth.shape)
    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss+1e-8)
    return loss

def lmae(coupling_preds, coupling_value):
    predict = coupling_preds.view(-1)
    truth   = coupling_value.view(-1)
    assert(predict.shape==truth.shape)
    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss+1e-8)
    return loss

# lmae for multi-type model 
def train_criterion(predict,
                    coupling_value, coupling_rank, coupling_contribution, coupling_type ,
                    criterion='lmae',
                    num_output= 1,
                    gaussrank=True,
                    pred_type= False): 
    '''
    The loss to be used for training the model w.r.t to flags: pred_type, num_output 
    
    TODO : Include per-type loss training 
    '''
    
    coupling_preds, contribution_preds, type_preds = predict 
    
    if not gaussrank: 
        coupling_rank = coupling_value
        
    # fix the regression loss to use : mse or lmae 
    if criterion == 'mse': 
        l = nn.MSELoss()
        
    elif criterion == 'lmae':
        l = lmae_criterion
        
    elif criterion == 'mlmae2ce':

        cross_entropy_loss = torch.nn.CrossEntropyLoss()(type_preds, coupling_type)
        abs_diff = torch.abs(coupling_preds - coupling_rank.view(-1,1).expand(coupling_preds.size()))

        if criterion == 'mse': 
            abs_diff = abs_diff**2

        proba_types = F.softmax(type_preds)
        weighted_diff = torch.mul(abs_diff, proba_types).sum(dim=1)

        unique_labels, labels_count = coupling_type.unique(dim=0, return_counts=True)
        res = torch.zeros(unique_labels.max()+1, dtype=torch.float, device='cuda')
        res = res.scatter_add_(0, coupling_type, weighted_diff)
        res = res[unique_labels]
        res = res.div(labels_count.float())
        res = res.log().mean()

        return  res  + 2 * cross_entropy_loss 

    elif criterion == 'mlmaeo2ce':
        cross_entropy_loss = torch.nn.CrossEntropyLoss()(type_preds, coupling_type)
        abs_diff = torch.abs(coupling_preds - coupling_rank.view(-1,1).expand(coupling_preds.size()))

        proba_types = F.softmax(type_preds)
        weighted_diff = torch.mul(abs_diff, proba_types).sum(dim=1)

        unique_labels, labels_count = coupling_type.unique(dim=0, return_counts=True)
        res = torch.zeros(unique_labels.max()+1, dtype=torch.float, device='cuda')
        res = res.scatter_add_(0, coupling_type, weighted_diff)
        res = res[unique_labels]
        res = res.div(labels_count.float())
        res = res.log().mean()

        if cross_entropy_loss >= 0.05:
            return  res  + 2 * cross_entropy_loss 
        else:
            return  res    

    elif criterion == 'mlmaeo2ceh':
        cross_entropy_loss = torch.nn.CrossEntropyLoss()(type_preds, coupling_type)
        abs_diff = torch.abs(coupling_preds - coupling_rank.view(-1,1).expand(coupling_preds.size()))

        proba_types = F.softmax(type_preds)
        
        weighted_diff = torch.sum((torch.index_select(abs_diff,1,torch.argmax(type_preds, dim=1))*torch.eye(len(coupling_type), device='cuda')), dim=1)

        unique_labels, labels_count = coupling_type.unique(dim=0, return_counts=True)
        res = torch.zeros(unique_labels.max()+1, dtype=torch.float, device='cuda')
        res = res.scatter_add_(0, coupling_type, weighted_diff)
        res = res[unique_labels]
        res = res.div(labels_count.float())
        res = res.log().mean()

        if cross_entropy_loss >= 0.05:
            return  res  + 2 * cross_entropy_loss 
        else:
            return  res 
        
    elif criterion == 'mlmaeo2ceha' or criterion == 'wmlmaeo2ceha':
        cross_entropy_loss = torch.nn.CrossEntropyLoss()(type_preds, coupling_type)
        abs_diff = torch.abs(coupling_preds - coupling_rank.view(-1,1).expand(coupling_preds.size()))

        proba_types = F.softmax(type_preds)
        
        weighted_diff = torch.sum((torch.index_select(abs_diff,1,coupling_type)*torch.eye(len(coupling_type), device='cuda')), dim=1)
        
        unique_labels, labels_count = coupling_type.unique(dim=0, return_counts=True)
        res = torch.zeros(unique_labels.max()+1, dtype=torch.float, device='cuda')
        res = res.scatter_add_(0, coupling_type, weighted_diff)
        if criterion == 'wmlmaeo2ceha':
            res = res * torch.tensor([10.,.1,.1,.1,.1,.1,.1,.1], dtype=torch.float, device='cuda')
        res = res[unique_labels]
        res = res.div(labels_count.float())
        if criterion == 'wmlmaeo2ceha':
            res = res*res
            res = res.mean()
        else:
            res = res.log().mean()

        if cross_entropy_loss >= 0.05:
            return  res  + 2 * cross_entropy_loss 
        else:
            return  res    
               
    elif criterion == 'lmaeo2ceha':
        cross_entropy_loss = torch.nn.CrossEntropyLoss()(type_preds, coupling_type)
        abs_diff = torch.abs(coupling_preds - coupling_rank.view(-1,1).expand(coupling_preds.size()))

        proba_types = F.softmax(type_preds)
        
        weighted_diff = torch.sum((torch.index_select(abs_diff,1,coupling_type)*torch.eye(len(coupling_type), device='cuda')), dim=1)
        
        res = torch.log(weighted_diff.mean())

        if cross_entropy_loss >= 0.05:
            return  res  + 2 * cross_entropy_loss 
        else:
            return  res   
        
    elif criterion == 'lmae_embed_type': 
        return lmae(coupling_preds, coupling_rank)
                              
    else: 
        raise Exception(f"""{criterion} is not handled""")
    
    if pred_type: 
        cross_entropy_loss = torch.nn.CrossEntropyLoss()(type_preds, coupling_type)
        abs_diff = torch.abs(coupling_preds - coupling_rank.view(-1,1).expand(coupling_preds.size()))
        
        if criterion == 'mse': 
            abs_diff = abs_diff**2
        
        proba_types = F.softmax(type_preds)
        weighted_diff = torch.mul(abs_diff, proba_types).sum(dim=1)
        weighted_loss = torch.log(weighted_diff.mean())
        
        weighted_loss =  weighted_loss + 2 * cross_entropy_loss
        return   weighted_loss
    
    elif num_output == 5: 
        loss_coupling = l(coupling_preds, coupling_rank)
        loss_fc =  l(contribution_preds[:, 0], coupling_contribution[:, 0])
        loss_sd =  l(contribution_preds[:, 1], coupling_contribution[:, 1])
        loss_pso =  l(contribution_preds[:, 2], coupling_contribution[:, 2])
        loss_dso =  l(contribution_preds[:, 3], coupling_contribution[:, 3])
        return   loss_coupling + (0.1 * (loss_fc + loss_sd + loss_pso + loss_dso) / 4)
    
    elif num_output ==1 :
        return l(coupling_preds, coupling_rank)
    
    else: 
        raise Exception(f"""{num_output} is not handled""")