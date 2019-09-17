from mpnn_model.common import *

from torch_scatter import *
from torch_geometric.utils import scatter_
import torch
import torch.nn as nn
import numbers

from mpnn_model.common_model import * 
from mpnn_model.regression_head import * 
from mpnn_model.message_passing import * 
from mpnn_model.RNN_attention import * 



__all__ = ['Net' ]


#############################################################################################################
#                                                                                                           #
#                                            END-to-END model                                               #
#                                                                                                           #
#############################################################################################################  
class Net(torch.nn.Module):
    def __init__(self,
                ConfigParams,
                y_range
                ):
        """
        Arguments: 
            mpnn:  Dictionary with all the needed arguments for GraphConv and Set2Set modules. 
            regression: Dictionary with all the needed arguments for regression output module. 
            batch_size: 
            num_target:
            predict_type:
        """
        super(Net, self).__init__()
        self.encoding = ConfigParams['model']['mpnn']['node_encoder']['encoding']
        self.num_output = ConfigParams['model']['regression']['num_output']
        self.predict_type = ConfigParams['model']['regression']['predict_type']
        self.y_range = y_range
        self.node_dim = ConfigParams['model']['mpnn']['edge_encoder']['node_dim']
        self.num_target = ConfigParams['model']['regression']['num_target']
        
        self.num_type = ConfigParams['model']['num_type']
        self.RNN = ConfigParams['model']['RNN']
        self.device = ConfigParams['train']['device']
        
        ###################-------------  MPNN representation ---------------####################
        self.num_propagate = ConfigParams['model']['mpnn']['T_steps']

        # Process the nodes features 
        self.preprocess = get_node_encoder(**ConfigParams['model']['mpnn']['node_encoder'])
        
        # Message 
        self.message_function = MessagePassing(ConfigParams)

        #Update 
        self.update_function = GRUUpdate(ConfigParams)
        
        #readout 
        self.readout  = Set2Set(**ConfigParams['model']['mpnn']['Set2Set'])
        
        ###################-------------  RNN representation ---------------####################
        if self.RNN: 
            self.rnn_attention = BI_RNN_Nodes(**ConfigParams['model']['node_seq'])
            self.default_node_vector = torch.distributions.uniform.Uniform(-1.0 / math.sqrt(self.node_dim),
                                                                       1.0 / math.sqrt(self.node_dim)).sample_n(self.node_dim)
            if self.device == 'cuda': 
                self.default_node_vector = self.default_node_vector.cuda()

        ###################---------------- Build predictions ------------------######################
        # embed type with one single output for all the types: 
        if self.num_target == 1 and not self.predict_type: 
            self.type_embedding =  nn.Embedding(16, 32, padding_idx=0)
        
        self.dense_layer, self.classify, self.predict, self.predicition_layers = get_regression_module(**ConfigParams['model']['regression'])
      
    
    def forward(self, 
                node,
                edge,
                edge_index,
                node_index,
                coupling_index,
                bond_type,
                x_atomic):
        
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        
        #--- Build the graph representation using MPNN 
        # Process nodes representation
        if self.encoding == 'one_hot':
            node   = self.preprocess(node)
 
        elif self.encoding == 'label': 
            node_cat, node_cont = node[:,:6].long(), node[:,-1].view(-1,1).float()
            node = self.preprocess(node_cat, node_cont) 
        
        # T-steps of message updates 
        for i in range(self.num_propagate):
            # node <- h_v^t
            messages = self.message_function(node, edge_index, edge, reuse_graph_tensors=(i != 0)) # m_v^t+1 = sum_w(E_vw * h_vw^t)
            node = self.update_function(messages, node)  # h_v^t+1 = GRU(m_v^t+1, h_v^t)

        # K-steps of readout function    
        pool = self.readout(node, node_index)
        

        if self.RNN: 
            #--- Get indices of the atoms  in the coupling shortest path
            num_coupling = len(coupling_index)
            coupling_atom0_index, coupling_atom1_index, coupling_atom2_index, coupling_atom3_index, coupling_type_index, coupling_batch_index = \
                torch.split(coupling_index,1,dim=1)
            # Concatenate the graph representation vecotr 'pool',
            pool  = torch.index_select( pool, dim=0, index=coupling_batch_index.view(-1))
            #pad random unseen node vector to node matrix 
            node = torch.cat([self.default_node_vector.view(1, -1), node], dim=0)
            # build node's embedding sequence 
            node0 = torch.index_select( node, dim=0, index=coupling_atom0_index.view(-1)+1).unsqueeze(1)
            node1 = torch.index_select( node, dim=0, index=coupling_atom1_index.view(-1)+1).unsqueeze(1)
            node2 = torch.index_select( node, dim=0, index=coupling_atom2_index.view(-1)+1).unsqueeze(1)
            node3 = torch.index_select( node, dim=0, index=coupling_atom3_index.view(-1)+1).unsqueeze(1)
            node_seq = torch.cat([node0, node1, node2, node3], dim=1) # bs x 4 x node_dim 
            # Get attention hidden states
            attention_node_seq = self.rnn_attention(node_seq, bond_type.view(-1, 4, 1), x_atomic.view(-1, 4, 1))
            
            
            # embed type 
            if not self.predict_type and self.num_type != 1: 
                coupling_type_embeds = self.type_embedding(coupling_type_index)
                
                input_regression = torch.cat([pool, attention_node_seq, coupling_type_embeds.view(-1, 32)],-1)
            
            else: 
                input_regression = torch.cat([pool, attention_node_seq],-1)
        
        else: 
            #--- Get indices of the coupling atoms 
            num_coupling = len(coupling_index)
            coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index,1,dim=1)
            #Concatenate the graph representation vecotr 'pool', the represetation vectors of the nodes : 
            #                         coupling_atom0 andcoupling_atom1 
            pool  = torch.index_select( pool, dim=0, index=coupling_batch_index.view(-1))
            node0 = torch.index_select( node, dim=0, index=coupling_atom0_index.view(-1))
            node1 = torch.index_select( node, dim=0, index=coupling_atom1_index.view(-1))
            input_regression = torch.cat([pool,node0,node1],-1)
            
            
        
        self.pool = pool 
        
        dense_representation = self.dense_layer(input_regression)
        
        #---Get the outputs : coupling_preds, contribution_preds, type_classes : 
        #w.r.t the two flags : num_output (1: scalar vs 5: scalar+contribution) &  predict_type: False (use the actual type) Vs True (predict the type)
        predict_type = []
        contribution_preds = []
        
        #--- Get the regression predictions w.r.t the coupling type 
        if self.num_output ==1:
            predict = self.predict(dense_representation)
            coupling_preds = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(predict) + self.y_range[0]
            
            # when the model predicts a vector w.r.t target type (8) : Additional condition on jointly predict the type or not 
            if self.num_target != 1: 
                if self.predict_type:
                    predict_type = self.classify(dense_representation)
                else: 
                    coupling_preds = torch.gather(predict, 1, coupling_type_index).view(-1)
                    coupling_preds = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(coupling_preds) + self.y_range[0]
                                           
        
        elif self.num_output==5:
            # get 5 dim prediction vector for each type : only works when num_targets = 8, not implemented for 
            if num_target==1: 
                raise LookupError('Predicting coupling contributions only implemented for multi-types model')
                
            preds = [self.predicition_layers[i](dense_representation).view(-1, 1, 5) for i in range(8)]
            predict = torch.cat(preds, dim=1)
            predict = torch.gather(predict, 1, coupling_type_index.view(-1, 1, 1).expand(predict.size(0), 1, 
                                                                                         predict.size(2))).squeeze()
            
            contribution_preds = predict[:,1:].view(-1, 4)
            coupling_preds = predict[:,0].view(-1)
            coupling_preds = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(coupling_preds) + self.y_range[0]
        
        return [coupling_preds, contribution_preds, predict_type]   
    