#
# coupling_cols = ['atom_index_0', 'atom_index_1','coupling_type','scalar_coupling',
#                  'gaussrank_coupling','fc','sd','pso','dso','id',]
#
# edge_cols :  ['atom_index_0', 'atom_index_1', 'edge_type', 'distance', 'angle' ]
#
# nodes cols : ['symbol','acceptor', 'donor', 'aromatic',  'hybridization', 'num_h', 'atomic']  
#
###################################################




from mpnn_model.common import *

import torch
from torch import _utils
from fastai.torch_core import to_device
import torch.nn.functional as F 

from fastai.basic_data import DataBunch
from fastai.basic_data import *
from fastai.tabular import *
from fastai import *

import copy

#EDGE_DIM   =  6
#NODE_DIM   = 13 ##  93  13
NUM_TARGET =  8  ## for 8 bond's types 
NODE_MAX, EDGE_MAX, COUPLING_MAX = 32, 816, 136

DATA_DIR = '/rapids/notebooks/srabhi/champs-2019/input'

# __all__ = ['TensorBatchDataset', 'tensor_collate', 'BatchGraphDataset', 'null_collate', 
#            'BatchDataLoader', '_BatchDataLoaderIter', 'BatchDataBunch' ]

#############################################################################################################
#                                                                                                           #
#                                               Load batch of tensors                                       #
#                                                                                                           #
#############################################################################################################

class BatchDataset(object):
    """An abstract class representing a Batch Dataset.
    All other datasets should subclass this. All subclasses should override
    ``__len__``, which provides the size of the dataset, ``__getitem__``,
    supporting integer indexing of batches in range from 0 to len(self)//batchsize exclusive,
    and ``shuffle`` which randomly shuffles the data, generally called per epoch.
    Batch datasets are meant to be iterated over in order rather than randomly accessed
    so the randomization has to happen first.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self):
        raise NotImplementedError

    def shuffle(self):
        raise NotImplementedError
#############################################################################################################
#                                                                                                           #
#                                             Batch dataset                                                 #
#                                                                                                           #
#############################################################################################################
class TensorBatchDataset(BatchDataset):
    """Batch Dataset wrapping Tensors.
        Args:
            *tensors (Tensor): tensors that have the same size of the first dimension.
                                        6 tensors are needed:
                                batch_node,  batch_edge, batch_coupling,
                                batch_num_node, batch_num_edge, batch_num_coupling         
            batch_size: The size of the batch to return
            pin_memory (bool, optional): If ``True``, the dataset will be pinned memory for faster copy to GPU.
            I saw no performance improvement to doing so but results may vary.
            COUPLING_MAX: dimension of molecule coupling features vector 
            mode: ['train', 'test']: when mode == 'test' return addition infor vector with coupling observations ids 
            csv:  ['train', 'test']: source of data 
       
       Method __getitem__ returns: 
                 2 modes:
                    'train' : (node, edge_feats, edge_index, node_index, coupling_index),  targets
                    'test'  : (node, edge_feats, edge_index, node_index, coupling_index), targets,  infor 
            It calls the collate function 'tensor_collate_old' in roder to : 
            - Remove padded values from the four tensors : batch_node, batch_edge, batch_coupling, batch_graussrank,
            - Re-arrange data into X / targets 
            - Create the index matrices:  edge_index, node_index, to keep track the variable sizes of molecule graphs.
    """

    def __init__(self, molecule_names, tensors, collate_fn, batch_size=1, pin_memory=False, COUPLING_MAX=136, mode = 'train', csv='train'):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors
        self.batch_size = batch_size
        self.num_samples = tensors[0].size(0)
        self.mode = mode 
        self.csv = csv
        self.molecule_names = molecule_names
        self.COUPLING_MAX = COUPLING_MAX
        self.collate_fn = collate_fn

        if pin_memory:
            for tensor in self.tensors:
                tensor.pin_memory()

    def __len__(self):
        if self.num_samples % self.batch_size == 0:
            return self.num_samples // self.batch_size
        else:
            return self.num_samples // self.batch_size + 1

    def __getitem__(self, item):
        idx = item * self.batch_size
        # Need to handle odd sized batches if data isn't divisible by batchsize
        if idx < self.num_samples and (
                idx + self.batch_size < self.num_samples or self.num_samples % self.batch_size == 0):
            batch_data = [tensor[idx:idx + self.batch_size] for tensor in self.tensors]

        elif idx < self.num_samples and idx + self.batch_size > self.num_samples:
            batch_data =  [tensor[idx:] for tensor in self.tensors]
        else:
            raise IndexError
        return self.collate_fn(batch_data, self.batch_size, self.COUPLING_MAX, self.mode)

    def __add__(self, tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert len(self.tensors) == len(tensors)
        assert all(self_tensor[0].shape == tensor[0].shape for self_tensor, tensor in zip(self.tensors, tensors))

        num_add_samples = tensors[0].size(0)
        self.num_samples = self.num_samples + num_add_samples
        self.tensors = [torch.cat((self_tensor, tensor)) for self_tensor, tensor in zip(self.tensors, tensors)]

    def shuffle_max(self): 
        num_nodes = self.tensors[4]  #num nodes 
        # sort tensors w.r.t the number of nodes in each molecule: Get larger ones first 
        sort_id = num_nodes.argsort(descending=True)
        # Compute the first batch
        first_batch_id = sort_id[:self.batch_size]
        # Shuffle the rest of indices 
        idx = sort_id[self.batch_size:][torch.randperm(self.num_samples-self.batch_size, dtype=torch.int64, device='cuda')]
        final_idx = torch.cat([first_batch_id, idx])
        #print(final_idx.shape)
        self.tensors = [tensor[final_idx] for tensor in self.tensors]
        
    def shuffle(self):
        idx = torch.randperm(self.num_samples, dtype=torch.int64, device='cuda')
        self.tensors = [tensor[idx] for tensor in self.tensors]

    def get_total_samples(self): 
        """
        Update total sample of dataset with the total number of coupling obs 
        Returns: 
            mask : the cv mask used to select a group of molecule names 
        """
        self.df = pd.read_csv(DATA_DIR + '/csv/%s.csv'%self.csv)
        mask = self.df['molecule_name'].isin(self.molecule_names)
        self.total_samples = self.df[mask].shape[0]
        return mask



#############################################################################################################
#                                                                                                           #
#                                                batch loader                                               #
#                                                                                                           #
#############################################################################################################
        
class BatchDataLoader(object):
    """Batch Data loader. Takes in a batch dataset and returns iterators that return whole batches of data.
    Arguments:
        dataset (BatchDataset): dataset from which to load the data.
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        device: str,  return batch data in the related device  (default: )
        
    """

    def __init__(self, batchdataset, shuffle=False, max_first=False, 
                 pin_memory=False, drop_last=False, device='cuda'):

        self.batch_size = batchdataset.batch_size
        self.dataset = batchdataset
        self.shuffle = shuffle
        self.max_first = max_first
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.device = device


    def __iter__(self):
        return _BatchDataLoaderIter(self)

    def __len__(self):
        if self.drop_last and self.dataset.num_samples%self.batch_size != 0:
            return len(self.dataset)-1
        else:
            return len(self.dataset)
    


    
class _BatchDataLoaderIter(object):
    """Iterates once over the BatchDataLoader's batchdataset, shuffling if requested"""
    def __init__(self, loader):
        self.batchdataset = loader.dataset
        self.batch_size = loader.batch_size
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.drop_last = loader.drop_last
        self.device = loader.device
        
        if loader.max_first: 
            self.batchdataset.shuffle_max()
            
        elif loader.shuffle:
            self.batchdataset.shuffle()

        self.idx = 0

    def __len__(self):
        if self.drop_last and self.batchdataset.num_samples%self.batch_size != 0:
            return len(self.batchdataset)-1
        else:
            return len(self.batchdataset)
         
    
    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        if self.batchdataset.mode == 'test':
            X, y, infor = self.batchdataset[self.idx]
            batch = (X, y)
        else: 
            batch = self.batchdataset[self.idx]
        # Note Pinning memory was ~10% _slower_ for the test examples I explored
        if self.pin_memory:
            batch = _utils.pin_memory.pin_memory_batch(batch)
        self.idx = self.idx+1
        
        # move the batch data to device 
        batch = to_device(batch, self.device)
        # return in the form of : xb,yb = (x_cat, x_cont), y
        if self.batchdataset.mode == 'test':
            return batch, infor
        return batch

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self


#############################################################################################################
#                                                                                                           #
#                                                Fastai DataBunch                                           #
#                                                                                                           #
#############################################################################################################

class BatchDataBunch(DataBunch):
    
    @classmethod
    def remove_tfm(cls, tfm:Callable)->None:
        "Remove `tfm` from `self.tfms`."
        if tfm in cls.tfms: cls.tfms.remove(tfm)
            
    @classmethod
    def add_tfm(cls,tfm:Callable)->None:
        "Add `tfm` to `self.tfms`."
        cls.tfms.append(tfm)

    
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=64, val_bs=None, 
                      num_workers:int=defaults.cpus, device:torch.device=None,
                      collate_fn:Callable=data_collate, tfms: List[Callable]=None, 
                       size:int=None, **kwargs)->'BatchDataBunch':
        
        
        cls.tfms = listify(tfms)
        
        
        val_bs = ifnone(val_bs, bs)
        
        datasets = [train_ds, valid_ds]
        
        if valid_ds is not None:
            cls.empty_val = False
        else:
            cls.empty_val = True
            
        datasets.append(test_ds)

        cls.device = defaults.device if device is None else device
        
        dls = [BatchDataLoader(d, shuffle=s, max_first=s,  pin_memory=False, drop_last=False, device=cls.device) for d,s in
               zip(datasets,(True,False,False)) if d is not None]

        cls.path = path 
        
        cls.dls = dls
    
        
        assert not isinstance(dls[0],DeviceDataLoader)
        
        
        # load batch in device 
        
        if test_ds is not None:
            cls.train_dl, cls.valid_dl, cls.test_dl = dls
        else: 
            cls.train_dl, cls.valid_dl = dls
            
            
        cls.path = Path(path)
        return cls

