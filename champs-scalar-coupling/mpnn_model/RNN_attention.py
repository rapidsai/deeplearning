from mpnn_model.common import *
import torch
import torch.nn as nn

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
        x_nodes [batch_size x 4 x node_dim] : sequence of nodes embeddings of the coupling's shortest path 
        x_coupling_type [batch_size x 4 x 1]:  sequence of in-coming bond type 
        X_atomic [batch_size x 4 x 1]: sequence of node's atomic number 
        '''
        
        x_type = self.type_encoder(x_coupling_type+1).squeeze() # +1 to encode padded/missing values to 0 
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
        