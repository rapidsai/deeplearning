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


#__all__ = ['LinearBn', 'MlpBn', 'CustomTabularModel', 'get_node_encoder', 'get_edge_encoder', 
#          'GraphConv', 'Set2Set', 'get_regression_module', 'Net' ]


#############################################################################################################
#                                                                                                           #
#                                            Base-line models                                               #
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
#                                         Node and edge encoders                                            #
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
        
        self.bias = nn.Parameter(torch.Tensor(self.node_dim)).cuda()
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
#                                   MPNN- PHASE2 : Readout function                                         #
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

#############################################################################################################
#                                                                                                           #
#                                   Nodes sequence : Attention-Bidirectional                                #
#                                                                                                           #
#############################################################################################################   

class BI_RNN_Nodes(torch.nn.Module):
    def attention_neuralnet(self, rnn_out, state):
        
        """
        #### credit to : https://github.com/wabyking/TextClassificationBenchmark
        """
        merged_state = torch.cat([s for s in state],1)  # merge the hidden states of the two directions 
        merged_state = merged_state.squeeze(0).unsqueeze(2)

        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)

        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2), weights 

    
    def __init__(self, 
                 node_dim,
                 hidden_size,
                 num_layers,
                 dropout,
                 batch_first,
                 bidirectional,
                 rnn_model='LSTM',
                 attention=True):
        
        
        super(BI_RNN_Nodes, self).__init__()
        
        self.type_encoder = nn.Embedding(16, 32, padding_idx=0)
        
        self.atomic_encoder = nn.Embedding(16, 32, padding_idx=0)

        
        self.attention = attention 
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size= node_dim + 64, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            batch_first=batch_first, bidirectional=bidirectional)
        else: 
            raise LookupError('only support LSTM ')
            
    def forward(self, x_nodes, x_coupling_type, x_atomic): 
        '''
        x_nodes [batch_size x path_length x node_dim] : sequence of nodes embeddings of the coupling's shortest path 
        x_coupling_type [batch_size x 4 x 1]:  sequence of in-coming bond type 
        '''
        
        x_type = self.type_encoder(x_coupling_type+1).squeeze()
        x_atomic = self.atomic_encoder(x_atomic+1).squeeze()
        
        x = torch.cat([x_nodes, x_type, x_atomic], dim=2)
       
        rnn_out, (final_hidden_state, final_cell_state) = self.rnn(x, None)
        
        if self.attention:
            last_tensor_item, weights = self.attention_neuralnet(rnn_out, final_hidden_state)
        else:
            # use mean instead of weighted attention 
            last_tensor = rnn_out[row_indices, :, :]
            last_tensor_item = torch.mean(last_tensor, dim=1)
        
        return last_tensor_item
        

#############################################################################################################
#                                                                                                           #
#                                            Output models                                                  #
#                                                                                                           #
#############################################################################################################  

def get_regression_module(num_output=1,
                          node_dim=128,
                          shared_layers=[1024, 512],
                          activation='relu',
                          dropout= 0., 
                          branch_layers=[], 
                          num_target=8,
                          predict_type =False): 
    '''
    Regression module 
    Outputs:  4 branches 
        dense_layer: shared branch that learns a dense representation from the concatenation of Graph representation, 
                     nodes of the coupling edge reprensentation. 
        classify: if predict_type==True, Classification branch that computes the logits of the 8 classes of coupling type
        predict:  if num_output==1,  Regression branch that computes  scalar coupling constant vector: 8 values (per type)
        predicition_layers: if num_output==5,  8 regression branches (one for each coupling types) that computes
                          the scalar coupling constant and the contribution components. 
    '''
    
    predicition_layers = []
    classify =[]
    predict = []
    dense_layer = LinearBn(node_dim*6, shared_layers[0], act=activation)
    
    if num_output==1: 
        predict = nn.Sequential(
            MlpBn(shared_layers[0], dimensions=shared_layers[1:], activation=activation, dropout=dropout),
            nn.Linear(shared_layers[-1], num_target)
        )
         
    elif num_output == 5: 
        model = nn.Sequential(
                MlpBn(shared_layers[0],
                      dimensions=branch_layers,
                      activation=activation,
                      dropout=dropout),
                nn.Linear(branch_layers[-1], num_output))
        predicition_layers  = nn.ModuleList([model for i in range(num_target)])
    
    if predict_type: 
        classify = nn.Sequential(
            LinearBn( shared_layers[0], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_target),)
        
    return dense_layer, classify, predict, predicition_layers


#############################################################################################################
#                                                                                                           #
#                                            END-to-END model                                               #
#                                                                                                           #
#############################################################################################################  
class Net(torch.nn.Module):
    def __init__(self,
                ConfigParams,
                num_target=8, 
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
        self.y_range = ConfigParams['model']['Classif']['y_range']
        self.node_dim = ConfigParams['model']['mpnn']['edge_encoder']['node_dim']
     
        
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

        ###################---------------- Build predictions ------------------######################
        
        self.rnn_attention = BI_RNN_Nodes(**ConfigParams['model']['node_seq'])
        
        self.dense_layer, self.classify, self.predict, self.predicition_layers = get_regression_module(**ConfigParams['model']['regression'])
          
        self.default_node_vector = torch.distributions.uniform.Uniform(-1.0 / math.sqrt(self.node_dim),
                                                                       1.0 / math.sqrt(self.node_dim)).sample_n(self.node_dim).cuda()
    
    def forward(self, 
                node,
                edge,
                edge_index,
                node_index,
                coupling_index,
                bond_type,
                x_atomic,):
        
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
        
        #--- Get indices of the coupling atoms 
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, coupling_atom2_index, coupling_atom3_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index,1,dim=1)
        
        #Concatenate the graph representation vecotr 'pool',

        pool  = torch.index_select( pool, dim=0, index=coupling_batch_index.view(-1))
        
        #pad random unseen node vector to node matrix 
        node = torch.cat([self.default_node_vector.view(1, -1), node], dim=0)
        
        # build node's embedding sequence 
        node0 = torch.index_select( node, dim=0, index=coupling_atom0_index.view(-1)+1).unsqueeze(1)
        node1 = torch.index_select( node, dim=0, index=coupling_atom1_index.view(-1)+1).unsqueeze(1)
        node2 = torch.index_select( node, dim=0, index=coupling_atom2_index.view(-1)+1).unsqueeze(1)
        node3 = torch.index_select( node, dim=0, index=coupling_atom3_index.view(-1)+1).unsqueeze(1)
        
        node_seq = torch.cat([node0, node1, node2, node3], dim=1) # bs x 4 x node_dim 
        
        #attention_node_seq = self.rnn_attention(node_seq, bond_type.view(-1, 4, 1), )
        attention_node_seq = self.rnn_attention(node_seq, bond_type.view(-1, 4, 1), x_atomic.view(-1, 4, 1))
        
        dense_representation = self.dense_layer(torch.cat([pool, attention_node_seq],-1))

        self.pool = pool 
        
        #--- Get the regression predictions w.r.t the coupling type 
        if self.num_output ==1:
            predict = self.predict(dense_representation)
        
        elif self.num_output==5:
            # get 5 dim prediction vector for each type : num_targets = 8 
            preds = [self.predicition_layers[i](dense_representation).view(-1, 1, 5) for i in range(8)]
            predict = torch.cat(preds, dim=1)
            
        #---Get the outputs : coupling_preds, contribution_preds, type_classes : 
        #w.r.t the two flags : num_output (1: scalar vs 5: scalar+contribution) &  predict_type: False (use the actual type) Vs True (predict the type)
        predict_type = []
        contribution_preds = []
        
        if self.predict_type: 
            predict_type = self.classify(dense_representation)
            if self.num_output == 5:
                contribution_preds = predict[:,:, 1:]
                coupling_preds = predict[:,:, 0]
            else: 
                coupling_preds = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(predict) + self.y_range[0]
                    
        elif self.num_output==1:
            coupling_preds =torch.gather(predict, 1, coupling_type_index).view(-1)
            coupling_preds = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(coupling_preds) + self.y_range[0]
                                   
        elif self.num_output==5:
            predict = torch.gather(predict, 1, coupling_type_index.view(-1, 1, 1).expand(predict.size(0), 1, 
                                                                                         predict.size(2))).squeeze()
            contribution_preds = predict[:,1:].view(-1, 4)
            coupling_preds = predict[:,0].view(-1)
            
        return [coupling_preds, contribution_preds, predict_type]   
