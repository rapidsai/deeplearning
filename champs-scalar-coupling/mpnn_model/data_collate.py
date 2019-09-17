import torch
import torch.nn.functional as F 

from mpnn_model.common import *
from mpnn_model.common_constants import * 
from mpnn_model.data import *

import copy

DATA_DIR = '/rapids/notebooks/srabhi/champs-2019/input'

__all__ = ['tensor_collate_rnn', 'tensor_collate_baseline']

def tensor_collate_rnn(batch, batch_size, COUPLING_MAX, mode='train'):
    """
    Function to apply dynamic padding of each batch in order to prepare inputs of the graph neural network 
    
    node, edge_feats, edge_index, node_index, coupling_index),  targets
    Returns: 
        X: 
           node [batch_size*molecules_num_nodes,  node_dim] : nodes states, variable size per batch 
           edge_feats [batch_size*molecules_num_edges,  edge_dim] : edges states, variable size per batch 
           edge_index [batch_size*molecules_num_edges,  2] : Index edges  of the same molecule
           node_index [batch_size*molecules_num_nodes,  1] : Index nodes of the same molecule 
           coupling_index
       Y: 
           targets [N_coupling, 4]: tuple of four targets (scalar_coupling, coupling_gaussrank, coupling_contribs, coupling_type)
       Test info: 
           infor:  the ids of the coupling needed for build submission files  # N_coupling,
    """
    
    batch_node, batch_edge, batch_coupling, batch_num_node, batch_num_edge, batch_num_coupling = batch
    batch_node = batch_node.reshape(-1, NODE_MAX, 7).float()
    batch_edge = batch_edge.reshape(-1, EDGE_MAX, 5)
    batch_coupling = batch_coupling.reshape(-1, COUPLING_MAX, 21)
    batch_size = batch_node.shape[0]
    
    #### Create nodes / edges / coupling masks : optimized V1 :  TO be optimized 
    mask = torch.cat([F.pad(torch.ones(i, device='cuda'), (0,NODE_MAX-i)).unsqueeze(0) for i in batch_num_node], dim=0)
    mask_coupling = torch.cat([F.pad(torch.ones(i, device='cuda'), (0,COUPLING_MAX-i)).unsqueeze(0) for i in batch_num_coupling], dim=0)
    mask_edge = torch.cat([F.pad(torch.ones(i, device='cuda'), (0,EDGE_MAX-i)).unsqueeze(0) for i in batch_num_edge], dim=0)

    #### Build the output X:
    # Get effective nodes / edges / coupling values : without padding 
    node = batch_node[mask.bool()].view(-1, 7)
    edge = batch_edge[mask_edge.bool()].view(-1, 5)
    coupling = batch_coupling[mask_coupling.bool()].view(-1, 21)
    
    # node indices track nodes of the same molecule 
    node_index = mask.nonzero()[:, 0]
    
    # Get edges feats and indices 
    edge_feats = edge[:, 2:]
    edge_index = edge[:, :2].long()
    
    # Get coupling path index 
    coupling_index = coupling[:, 10:14].long()
    num_coupling = coupling_index.shape[0]
    
    #get sequence of coupling type 
    pad_vector = torch.zeros(num_coupling, device='cuda').long()-1
    coupling_type_sequence =  torch.cat([pad_vector.view(-1,1), coupling[:, 14:17].long()], 1)
    
    
    # batch_coupling_index : track coupling values of the same molecule 
    batch_coupling_index = mask_coupling.nonzero()[:, 0]
    
    # offset edge and coupling indices w.r.t to N of nodes in each molecule 
    offset = torch.cat([torch.zeros(1, device='cuda').long(), batch_num_node[:-1]]).cumsum(0)
    #edge 
    expanded_offset = torch.cat([torch.zeros(num_edges, device='cuda')+offset[mol_index] for mol_index,num_edges in 
                                 enumerate(batch_num_edge)], 0)
    edge_index = torch.add(edge_index, expanded_offset.unsqueeze(1).long())
    #coupling 
    expanded_offset=torch.cat([torch.zeros(n_coupling, device='cuda')+offset[mol_index] for mol_index,n_coupling in 
                               enumerate(batch_num_coupling)], 0)
    coupling_index = torch.add(coupling_index, expanded_offset.unsqueeze(1).long())
    # type_id 
    coupling_type = coupling[:, 2].long()

    # new coupling index: atom_0, atom_1, atom_2, atom_3, coupling_type, batch_index
    coupling_index = torch.cat([coupling_index, coupling_type.view(-1,1) , batch_coupling_index.view(-1, 1)], -1)
    
    # get sequence of atomic number 
    coupling_atomic = coupling[:, 17:].long()
        
    #### Get Targets 
    # 4 coupling contirbutions 
    coupling_contribution = coupling[:, 5:9]
    #coupling value 
    coupling_value = coupling[:, 3]
    #gauss_rank 
    gaussrank = coupling[:, 4]
    targets = [coupling_value.float(), gaussrank.float(), coupling_contribution.float(), coupling_type.long()] 
    
    #### ids for inference time 
    infor = coupling[ : , 9]

    # mode flag to return additional information for test data 
    if mode == 'test':
            return (node, edge_feats, edge_index, node_index, coupling_index, coupling_type_sequence, coupling_atomic), targets,  infor 

    return (node, edge_feats, edge_index, node_index, coupling_index, coupling_type_sequence, coupling_atomic),  targets

def tensor_collate_baseline(batch, batch_size, COUPLING_MAX, mode='train'):
    """
    Function to apply dynamic padding of each batch in order to prepare inputs of the graph neural network 
    
    node, edge_feats, edge_index, node_index, coupling_index),  targets
    Returns: 
        X: 
           node [batch_size*molecules_num_nodes,  node_dim] : nodes states, variable size per batch 
           edge_feats [batch_size*molecules_num_edges,  edge_dim] : edges states, variable size per batch 
           edge_index [batch_size*molecules_num_edges,  2] : Index edges  of the same molecule
           node_index [batch_size*molecules_num_nodes,  1] : Index nodes of the same molecule 
           coupling_index
       Y: 
           targets [N_coupling, 4]: tuple of four targets (scalar_coupling, coupling_gaussrank, coupling_contribs, coupling_type)
       Test info: 
           infor:  the ids of the coupling needed for build submission files  # N_coupling,
    """
    
    batch_node, batch_edge, batch_coupling, batch_num_node, batch_num_edge, batch_num_coupling = batch
    batch_node = batch_node.reshape(-1, NODE_MAX, 7).float()
    batch_edge = batch_edge.reshape(-1, EDGE_MAX, 5)
    batch_coupling = batch_coupling.reshape(-1, COUPLING_MAX, 10)
    batch_size = batch_node.shape[0]
    
    #### Create nodes / edges / coupling masks : optimized V1 :  TO be optimized 
    mask = torch.cat([F.pad(torch.ones(i, device='cuda'), (0,NODE_MAX-i)).unsqueeze(0) for i in batch_num_node], dim=0)
    mask_coupling = torch.cat([F.pad(torch.ones(i, device='cuda'), (0,COUPLING_MAX-i)).unsqueeze(0) for i in batch_num_coupling], dim=0)
    mask_edge = torch.cat([F.pad(torch.ones(i, device='cuda'), (0,EDGE_MAX-i)).unsqueeze(0) for i in batch_num_edge], dim=0)

    #### Build the output X:
    # Get effective nodes / edges / coupling values : without padding 
    node = batch_node[mask.bool()].view(-1, 7)
    edge = batch_edge[mask_edge.bool()].view(-1, 5)
    coupling = batch_coupling[mask_coupling.bool()].view(-1, 10)
    
    # node indices track nodes of the same molecule 
    node_index = mask.nonzero()[:, 0]
    
    # Get edges feats and indices 
    edge_feats = edge[:, 2:]
    edge_index = edge[:, :2].long()
    
    # Get coupling index 
    coupling_index = coupling[:, :2].long()
    # batch_coupling_index : track coupling values of the same molecule 
    batch_coupling_index = mask_coupling.nonzero()[:, 0]
    
    # offset edge and coupling indices w.r.t to N of nodes in each molecule 
    offset = torch.cat([torch.zeros(1, device='cuda').long(), batch_num_node[:-1]]).cumsum(0)
    #edge 
    expanded_offset = torch.cat([torch.zeros(num_edges, device='cuda')+offset[mol_index] for mol_index,num_edges in 
                                 enumerate(batch_num_edge)], 0)
    edge_index = torch.add(edge_index, expanded_offset.unsqueeze(1).long())
    #coupling 
    expanded_offset=torch.cat([torch.zeros(n_coupling, device='cuda')+offset[mol_index] for mol_index,n_coupling in 
                               enumerate(batch_num_coupling)], 0)
    coupling_index = torch.add(coupling_index, expanded_offset.unsqueeze(1).long())
    # type_id 
    coupling_type = coupling[:, 2].long()

    # new coupling index: atom_0, atom_1, coupling_type, batch_index
    coupling_index = torch.cat([coupling_index, coupling_type.view(-1,1) , batch_coupling_index.view(-1, 1)], -1)
    
        
    #### Get Targets 
    # 4 coupling contirbutions 
    coupling_contribution = coupling[:, 5:9]
    #coupling value 
    coupling_value = coupling[:, 3]
    #gauss_rank 
    gaussrank = coupling[:, 4]
    targets = [coupling_value.float(), gaussrank.float(), coupling_contribution.float(), coupling_type.long()] 
    
    #### ids for inference time 
    infor = coupling[ : , 9]

    
    # We don't use sequence data of the shortest path 
    coupling_type_sequence, coupling_atomic = [], []
    # mode flag to return additional information for test data 
    if mode == 'test':
            return (node, edge_feats, edge_index, node_index, coupling_index, coupling_type_sequence, coupling_atomic), targets,  infor 

    return (node, edge_feats, edge_index, node_index, coupling_index, coupling_type_sequence, coupling_atomic),  targets

