import os
from datetime import datetime
from functools import partial
from timeit import default_timer as timer
from time import time 
import warnings
warnings.filterwarnings("ignore") 

from mpnn_model.common import *
from mpnn_model.common_constants import *
from mpnn_model.train_loss import lmae_criterion
from mpnn_model.callback import * 

from time import time 
#############################################################################################################
#                                                                                                           #
#                                     Get prediction function                                               #
#                                                                                                           #
#############################################################################################################
def do_test(net, test_loader, test_len, num_output, predict_type, grm, normalize=False, gaussrank=False):
    """
    do_test -> return list of (indices, predictions)
    
    Input arguments: 
        net (nn.module) : the graph neural network model 
        test_loader (Dataloader):  Test data loader
        test_len (int):  length of test dataset     
    """
    
    test_num = 0
    test_predict = []
    test_coupling_type  = []
    test_coupling_value = []
    test_id = []
    test_contributions = []
    molecule_representation = []
    num_batches = 0
    test_loss = 0 
    start = timer()    
    for b, (((node, edge, edge_index, node_index, coupling_index, type_, atomic), targets), infor) in enumerate(test_loader):
        net.eval()
        with torch.no_grad():
            
            coupling_value = targets[0]
            predict =  net(node, edge, edge_index, node_index, coupling_index, type_, atomic)
            
            if predict_type: 
                predict = torch.gather(predict[0], 1, targets[3].unsqueeze(1)).view(-1)
                predict = [predict, [], []]
                
            if normalize: 
                coupling_mean = torch.gather(COUPLING_TYPE_MEAN, 0, targets[3])
                coupling_std = torch.gather(COUPLING_TYPE_STD, 0, targets[3])
                predict = (predict * coupling_std) + coupling_mean
                coupling_value = (coupling_value * coupling_std) + coupling_mean
                predict = [predict, [], []]
                
            loss = lmae_criterion(predict, coupling_value, coupling_value, [], [])
                
                
        batch_size = test_loader.batch_size
        test_id.extend(list(infor.data.cpu().numpy()))
        
        test_predict.append(predict[0].data.cpu().numpy())
        molecule_representation.append(net.pool.data.cpu().numpy())
        
        test_coupling_type.append(coupling_index[:,-2].data.cpu().numpy())
        test_coupling_value.append(coupling_value.data.cpu().numpy())

        test_loss += loss.item()*batch_size
        test_num = len(test_id)
        num_batches += batch_size 
        
        print('\r %8d/%8d     %0.2f  %s'%( test_num, test_len, test_num/test_len,
                                          time_to_str(timer()-start,'min')),end='',flush=True)
        
        pass
    
    test_loss = test_loss/num_batches
    print('\n')
    
    print('predict')
    predict  = np.concatenate(test_predict)
    
    if num_output==5:
        contributions = np.concatenate(test_contributions)
    else:
        contributions = []
        
    test_coupling_value = np.concatenate(test_coupling_value)
    test_coupling_type  = np.concatenate(test_coupling_type).astype(np.int32)
    molecule_representation = np.concatenate(molecule_representation)
    
    # convert gaussrank test predictions to their actual values 
    if gaussrank: 
        print('compute the reverse frame')
        reverse_frame = get_reverse_frame(test_id, predict, test_coupling_type, test_coupling_value, grm)
        predict = reverse_frame['scalar_coupling_constant'].values
    
    else: 
        print('build preds frame')
        reverse_frame = pd.DataFrame(predict)
        reverse_frame['type'] = test_coupling_type
        reverse_frame.columns = ['scalar_coupling_constant', 'type_ind']
        reverse_frame['id'] = test_id
        reverse_frame['true_scalar_coupling_constant'] = test_coupling_value
        
    
    mae, log_mae   = compute_kaggle_metric(reverse_frame.scalar_coupling_constant, reverse_frame.true_scalar_coupling_constant, reverse_frame.type_ind)
    
    print('Compute lmae per type')
    num_target = NUM_COUPLING_TYPE
    for t in range(NUM_COUPLING_TYPE):
         if mae[t] is None:
            mae[t] = 0
            log_mae[t]  = 0
            num_target -= 1

    mae_mean, log_mae_mean = sum(mae)/NUM_COUPLING_TYPE, sum(log_mae)/NUM_COUPLING_TYPE
    
    loss_ = log_mae + [ test_loss, mae_mean, log_mae_mean ]
    
    return loss_, reverse_frame, contributions, molecule_representation        