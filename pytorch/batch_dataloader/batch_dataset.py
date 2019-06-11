import torch

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

        
        
class TensorBatchDataset(BatchDataset):
    """Batch Dataset wrapping Tensors.  
    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
        batch_size: The size of the batch to return
        pin_memory (bool, optional): If ``True``, the dataset will be pinned memory for faster copy to GPU.  
        I saw no performance improvement to doing so but results may vary.        
    """
    def __init__(self, tensors, batch_size=1, pin_memory=False):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors      
        self.batch_size=batch_size
        
        self.num_samples = tensors[0].size(0)
        
        if pin_memory:
            for tensor in self.tensors:
                tensor.pin_memory() 
    
    def __len__(self):
        if self.num_samples%self.batch_size == 0:
            return self.num_samples // self.batch_size
        else:
            return self.num_samples // self.batch_size + 1

    def __getitem__(self, item):
        idx = item*self.batch_size
        #Need to handle odd sized batches if data isn't divisible by batchsize
        if idx < self.num_samples and (idx + self.batch_size < self.num_samples or self.num_samples%self.batch_size == 0):
            return [tensor[idx:idx+self.batch_size] for tensor in self.tensors]
        elif idx < self.num_samples and idx + self.batch_size> self.num_samples :
            return [tensor[idx:] for tensor in self.tensors]
        else:
            raise IndexError
        
    def __add__(self, tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert len(self.tensors) == len(tensors)
        assert all(self_tensor[0].shape == tensor[0].shape for self_tensor, tensor in zip(self.tensors, tensors))
        
        num_add_samples = tensors[0].size(0)
        self.num_samples = self.num_samples + num_add_samples
        self.tensors  = [torch.cat((self_tensor, tensor)) for self_tensor, tensor in zip(self.tensors, tensors)]
    
    def shuffle(self):
        idx = torch.randperm(self.num_samples, dtype=torch.int64)
        self.tensors = [tensor[idx] for tensor in self.tensors]
