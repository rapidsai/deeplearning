import numpy as np 
import pandas as pd 

from fastai.callbacks import SaveModelCallback
from fastai.callbacks import Callback
from fastai.torch_core import add_metrics

import torch 
from torch import nn 
import torch.nn.functional as F

import pdb

from mpnn_model.common_constants import NUM_COUPLING_TYPE, COUPLING_TYPE_MEAN, COUPLING_TYPE_STD, REVERSE_COUPLING_TYPE


__all__ = ['get_reverse_frame', 'lmae', 'compute_kaggle_metric', 'LMAE']

# reverse the gaussrank predictions to the actual distribution
def get_reverse_frame(test_id, predictions, coupling_type, target, grm,):
    preds = pd.DataFrame(predictions)
    preds['type_ind'] = coupling_type
    preds.columns = ['prediction', 'type_ind']
    preds['type'] = preds['type_ind'].map(REVERSE_COUPLING_TYPE)
    preds['id'] = test_id
    preds['true_scalar_coupling_constant'] = target
    preds['scalar_coupling_constant'] = grm.convert_df(preds, from_coupling=False)
    return preds 


# Compute lmae of scalar coupling with respect to the type `
def lmae(truth,pred,types):
    # Log of the Mean Absolute Error
    # will make it per type later
    df = pd.DataFrame({'truth':truth,'pred':pred,'types':types})
    df['err'] = np.abs(df['truth']-df['pred'])
    x = df.groupby('types')['err'].mean().values
    x = np.log(1e-8+x)
    return np.mean(x)   


#  lmae w.r.t  8 coupling types : kaggle metric 
def compute_kaggle_metric(predict, coupling_value, coupling_type):
    """
    predict lmae loss w.r.t the coupling type 
    
    Arguments: 
        -  predict: type(Array) array of scalar coupling predictions returned by the model  
        -  coupling_value: type(Array )  True coupling values 
        -  coupling_type: type(Array)  True coupling type 
    Returns: 
        - mae, log_mae : the mean and log mean absolute error between predictions and true labels 
    """

    mae     = [None]*NUM_COUPLING_TYPE
    log_mae = [None]*NUM_COUPLING_TYPE
    diff = np.fabs(predict-coupling_value)
    for t in range(NUM_COUPLING_TYPE):
        index = np.where(coupling_type==t)[0]
        if len(index)>0:
            m = diff[index].mean()
            log_m = np.log(m+1e-8)

            mae[t] = m
            log_mae[t] = log_m
        else:
            pass
    return mae, log_mae


# Callback to calculate LMAE at the end of each epoch
class LMAE(Callback):
    '''
    Comput LMAE for the prediction of the coupling value 
    '''
    _order = -20 #Needs to run before the recorder

    def __init__(self, learn,grm, predict_type=False, normalize_coupling=False, coupling_rank=True, **kwargs): 
        self.learn = learn
        self.predict_type = predict_type 
        self.normalize_coupling = normalize_coupling
        self.grm = grm 
        self.coupling_rank = coupling_rank 
    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['LMAE'])
    def on_epoch_begin(self, **kwargs): self.output, self.target, self.types = [], [], []

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.target.append(last_target[0])
            self.types.append(last_target[3])
            if self.predict_type: 
                coupling = torch.gather(last_output[0], 1, last_target[3].unsqueeze(1)).view(-1)
                self.output.append(coupling)
            else:
                self.output.append(last_output[0])
            
    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            types = torch.cat(self.types)
            
            if self.normalize_coupling : 
                # Denormalize w.r.t to type 
                means = torch.gather(COUPLING_TYPE_MEAN, 0, types)
                stds = torch.gather(COUPLING_TYPE_STD, 0, types)
                output = (output * stds) + means
                target = (target * stds) + means 
                metric = lmae(output.data.cpu().numpy(), target.data.cpu().numpy(), types.data.cpu().numpy())
                
            elif self.coupling_rank: 
                # Reverse using grm mapping frames 
                preds = pd.DataFrame(output.data.cpu().numpy())
                preds['type'] = types.data.cpu().numpy()
                preds.columns = ['prediction', 'type']
                preds['type'] = preds['type'].map(REVERSE_COUPLING_TYPE)
                preds['true_scalar_coupling_constant'] = target.data.cpu().numpy()
                preds['scalar_coupling_constant'] = self.grm.convert_df(preds, from_coupling=False) 
                
                # compute metric for reversed scalar coupling predictions  
                metric = lmae(preds['scalar_coupling_constant'], preds['true_scalar_coupling_constant'], preds['type'])
                
            else: 
                preds = output.data.cpu().numpy().reshape(-1,)
                type_ = types.data.cpu().numpy()
                targets =  target.data.cpu().numpy()
                metric = lmae(targets, preds, type_)
                
            return add_metrics(last_metrics, [metric])
