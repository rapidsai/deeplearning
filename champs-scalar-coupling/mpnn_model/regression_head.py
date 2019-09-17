from mpnn_model.common import *
from mpnn_model.common_model import * 

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['get_regression_module']

#############################################################################################################
#                                                                                                           #
#                                            Output models                                                  #
#                                                                                                           #
#############################################################################################################  

def get_regression_module(num_output=1,
                          input_dim=128,
                          shared_layers=[1024, 512],
                          activation='relu',
                          dropout= 0., 
                          branch_layers=[], 
                          num_target=8,
                          predict_type =False): 
    '''
    Regression module 
    Args: 
        num_output: [1, 5]: Whether to predict only scalar coupling or scalar coupling + the 4 contributions 
        input_dim: the dimension of regression's head input: 
                Combination of Graph representation, nodes' reprensentation of the coupling edge, nodes sequence hidden states.
        shared_layers: the dimension of the fully connected network shared between all the possible model's outputs
        activation: 
        dropout: probability dropout for regresson regularization 
        branch_layers: the fully connected branch network to predict each contribution value 
        num_target: Whether to predict all the coupling type or fine-tune a single model per type 
        predict_type: For num_output =1, whether to jointly predict the bond type or to embed the categorical variable "bond type"
    
    Outputs:  4 branches 
        dense_layer: shared branch that learns a dense representation from the concatenation of a combination of 
                        Graph representation, nodes' reprensentation of the coupling edge, . 
        classify: if predict_type==True, Classification branch that computes the logits of the 8 classes of coupling type
        predict:  if num_output==1,  Regression branch that computes  scalar coupling constant vector: 8 values (per type)
        predicition_layers: if num_output==5,  8 regression branches (one for each coupling types) that computes
                          the scalar coupling constant and the contribution components. 
    '''
    
    predicition_layers = []
    classify =[]
    predict = []
    dense_layer = LinearBn(input_dim, shared_layers[0], act=activation)
    
    ### Whether to predict only scalar coupling or scalar coupling + the 4 contributions 
    if num_output==1: 
        predict = nn.Sequential(
            MlpBn(shared_layers[0], dimensions=shared_layers[1:], activation=activation, dropout=dropout),
            nn.Linear(shared_layers[-1], num_target)
        )
        ### Whether to jointly predict the bond type or to embed the categorical variable "bond type"
        if predict_type: 
            classify = nn.Sequential(
                LinearBn( 1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_target),)
         
    elif num_output == 5: 
        model = nn.Sequential(
                MlpBn(shared_layers[0],
                      dimensions=branch_layers,
                      activation=activation,
                      dropout=dropout),
                nn.Linear(branch_layers[-1], num_output))
        predicition_layers  = nn.ModuleList([model for i in range(num_target)])
        
    return dense_layer, classify, predict, predicition_layers
