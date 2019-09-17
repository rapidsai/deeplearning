from mpnn_model.common_model import * 
from mpnn_model.common import * 

from torch_scatter import *
from torch_geometric.utils import scatter_
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers


__all__ = ['message_pass' , 'MessagePassing', 'GRUUpdate', 'Set2Set']

#############################################################################################################
#                                                                                                           #
#                               MPNN- PHASE1 : Message Passing                                              #
#                                                                                                           #
#############################################################################################################   
def message_pass(node_states, edge_index,  a_in):
    """Computes a_t from h_{t-1}, see bottom of page 3 in the paper.
                a_t = sum_w A(e_vw) . h^t
      Args:
        node_states: [batch_size*num_nodes, node_dim] tensor (h_{t-1})
        a_in (torch.float32): [batch_size*num_nodes, node_dim,  node_dim]:  Encoded edge matrix 
        edge_index [batch_size*num_edges, 2]: the indices of edges 
      Returns:
        messages (torch.float32): [batch_size*num_nodes, node_dim] For each pair
          of nodes in the graph a message is sent along both the incoming edge.
    """
    num_node, node_dim = node_states.shape
    edge_index = edge_index.t().contiguous()
    x_i  = torch.index_select(node_states, 0, edge_index[0])
    message = torch.matmul( x_i.view(-1,1,node_dim), a_in).view(-1, node_dim)
    message = scatter_('mean', message, edge_index[1], dim_size=num_node)
    return message

class MessagePassing(nn.Module):
    '''
    A feed forward neural network is applied to each edge in the adjacency matrix,
    which is assumed to be vector valued. It maps the edge vector to a
    node_dim x node_dim matrix, denoted NN(e). The message from node v -> w is
    then NN(e) h_v. This is a generalization of the message function in the
    GG-NN paper, which embeds the discrete edge label as a matrix.
    '''
    def __init__(self, ConfigParams):
        '''

        '''
        super(MessagePassing, self).__init__()
        self.encoding =  ConfigParams['model']['mpnn']['node_encoder']['encoding']
        self.edge_encoder = get_edge_encoder(**ConfigParams['model']['mpnn']['edge_encoder'])
        self.node_dim = ConfigParams['model']['mpnn']['edge_encoder']['node_dim']
        self.device = ConfigParams['train']['device']

        if self.device == 'cuda':
            self.bias = nn.Parameter(torch.Tensor(self.node_dim)).cuda()
        else: 
            self.bias = nn.Parameter(torch.Tensor(self.node_dim))
            
            
        self.bias.data.uniform_(-1.0 / math.sqrt(self.node_dim), 1.0 / math.sqrt(self.node_dim))

        self._a_in = [] 
     
    def _pre_encode_edges(self, edge):
        '''
        Args: 
        edge:  [batch_size*num_edges, edge_dim] edge features
        Return: 
            A neural representation of the edge festures where each vector is represented as 
            matrix of shape node_dim x node_dim 
        '''
        if self.encoding == 'label':
            edge_cat = edge[:, 0].long().view(-1,1)
            edge_cont = edge[:, 1:].float()
            edge = self.edge_encoder(edge_cat, edge_cont).view(-1,self.node_dim,self.node_dim)

        elif self.encoding == 'one_hot': 
            edge    = self.edge_encoder(edge).view(-1, self.node_dim, self.node_dim)
        
        self._a_in = edge 
        
    def forward(self, node_states, edge_index, edge, reuse_graph_tensors=True): 
        '''
        Args:
                node_states: [batch_size*num_nodes, node_dim] tensor (h_{t-1})
                edge_in: [batch_size*num_nodes, edge_dim] (torch.int32)
                reuse_graph_tensors: Boolean to indicate whether or not the self._a_in
                    should be reused or not. Should be set to False on first call, and True
                    on subsequent calls.
        Returns:
                message_t: [batch_size * num_nodes, node_dim] which is the node representations
                after a single propgation step
        '''
        if not reuse_graph_tensors:
            self._pre_encode_edges(edge)
        new_state = message_pass(node_states, edge_index, self._a_in)
        return  F.relu(new_state + self.bias)
        
#############################################################################################################
#                                                                                                           #
#                                 MPNN- PHASE2 : Updage nodes states                                        #
#                                                                                                           #
#############################################################################################################  

class GRUUpdate(nn.Module): 
    def __init__(self, ConfigParams):
        super(GRUUpdate, self).__init__()
        
        self.node_dim = ConfigParams['model']['mpnn']['edge_encoder']['node_dim'] 
        self.gru  = nn.GRU(self.node_dim, self.node_dim, batch_first=False, bidirectional=False)

    def forward(self, messages, node_states):
        """Build the fprop graph.
        Args:
            node_states: [batch_size*num_nodes, node_dim] tensor (h_{t-1})
            messages: [batch_size*num_nodes, node_dim] (a_t from the GGNN paper)
        Returns:
            updated_states: [batch_size*num_nodes, node_dim]
        """
        num_node, node_dim = node_states.shape

        update, _ = self.gru(messages.view(1,-1,self.node_dim),
                                node_states.view(1,num_node,-1))

        return  update.view(-1,node_dim)
    
#############################################################################################################
#                                                                                                           #
#                                   MPNN- PHASE3 : Readout function                                         #
#                                                                                                           #
#############################################################################################################   

class Set2Set(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1, num_layer = 1, batch_size=32):
        super(Set2Set, self).__init__()
        
        out_channel = 2 * in_channel
        self.processing_step = processing_step
        self.batch_size = batch_size
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_layer   = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()


    def forward(self, x, batch_index):
        h = (x.new_zeros((self.num_layer, self.batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, self.batch_size, self.in_channel)))
        # zeros of shape:  bs x 2*node_dim : init q_star 
        q_star = x.new_zeros(self.batch_size, self.out_channel)

        # n readout steps 
        for i in range(self.processing_step): 
            # read from memory 
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(self.batch_size, -1)
            #energies : dot product between input_set and q 
            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            # Compute attention  
            a = self.softmax(e, batch_index, num=self.batch_size)   #shape = num_node x 1
            #compute readout            
            r = scatter_add(a * x, batch_index, dim=0, dim_size=self.batch_size) #apply attention #shape = batch_size x ...
            #update q_star
            q_star = torch.cat([q, r], dim=-1)
            # print(q_star.shape)
        return q_star

