from mpnn_model.common import *

from torch_scatter import *
from torch_geometric.utils import scatter_
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

# Fast ai
from fastai.tabular import *
from fastai.callbacks import SaveModelCallback


__all__ = ['LinearBn', 'MlpBn', 'CustomTabularModel', 'get_node_encoder', 'get_edge_encoder',]


#############################################################################################################
#                                                                                                           #
#                                           Linear batch-norm layers                                        #
#                                                                                                           #
#############################################################################################################

class LinearBn(nn.Module):
    '''
    Batch norm dense layer  
    Arguments: 
        - in_channel: int,   Input dimension 
        - out_channel: int,  Output dimension 
        - act:  str, Activation function to apply to the output of batch normalizaiton. 
    '''
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel, eps=1e-05, momentum=0.1)
        if act is not None :
            self.act  =  F.__dict__[act]
        else: 
            self.act = act 

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
    
    
class MlpBn(nn.Module):
    ''' Fully connected feed forward neural network: stacked batch norm layers with dropout 
    Args:
        input_dim (int32): the dimension of input 
        dimensions (int32): the dimension of  hiddenlayers. 
        act (string): Activation function to apply to the output of each layer. 
        dropout (float): the dropout probabily to apply to each layer. 
    '''
    def __init__(self,
            input_dim,
            dimensions,
            activation='Relu',
            dropout=0.):
        
        super(MlpBn, self).__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout

        # Modules
        self.linears = nn.ModuleList([LinearBn(input_dim, dimensions[0], act=activation)])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(LinearBn(din, dout, act=self.activation))
    
    def forward(self, x):
        for i,lin in enumerate(self.linears):
            x = lin(x)
            if self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)
        return x

#############################################################################################################
#                                                                                                           #
#                                                 Tabular model                                             #
#                                                                                                           #
#############################################################################################################
    
class CustomTabularModel(nn.Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        #self.bsn = BatchSwapNoise(0.15)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        layers = layers[:-2]
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        #self.bsn(x_cat)

        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        return x
    
    
#############################################################################################################
#                                                                                                           #
#                                         Node and edge encoders                                            #
#                                                                                                           #
#############################################################################################################
    
def get_node_encoder(encoding, emb_sz, n_cont, node_dim, layers, activation, dropout=0.):
    '''
    - Get the MLP network to process nodes features and build node representation
    '''
    if encoding == 'one_hot':
        return MlpBn(node_dim, dimensions=layers, activation=activation, dropout=dropout)
    elif encoding== 'label': 
        # embed symbol, acceptor, donor, aromatic, hybridization 
        # emb_sz = [(6,4), (3,3), (3,3), (3,3), (5,4)]
        return CustomTabularModel(emb_szs = emb_sz, out_sz=2, n_cont=n_cont, layers=layers, ps=[dropout], emb_drop=0.)


def get_edge_encoder(encoding, emb_sz, n_cont,  node_dim, edge_dim,  layers, activation, dropout=0.):
    '''
    Get the MLP network to process edges features and build matrix representation
    Arguments:
        - encoding: str, the encoding of categorical variables : "label" vs "one_hot"
        - emb_sz: list of tuples,  the embedding size of each categorical variable
        - n_cont:  int, the number of continious variables 
        - node_dim: int, the dimension of node's representation 
        - edge_dim: int, the input dimension of edge's features 
        - layers: list of int, the dimensions of hidden layers 
        - activation: str,  the activation to apply for layers. 
        - dropout: [float],   dropout of each hidden layer. 
    '''
    if encoding == 'one_hot':
        return  MlpBn(edge_dim, dimensions=layers+[node_dim*node_dim], activation=activation, dropout=dropout) 

    elif encoding== 'label': 
        # emb_sz = [(5,8)]
        return CustomTabularModel(emb_szs = emb_sz, n_cont=n_cont , out_sz=2, layers=layers+[node_dim*node_dim], ps=[dropout], emb_drop=0.)
