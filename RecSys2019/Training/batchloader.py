import torch
from torch import _utils
from fastai.torch_core import to_device

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
        self.batch_size = batch_size

        self.num_samples = tensors[0].size(0)

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
            return [tensor[idx:idx + self.batch_size] for tensor in self.tensors]

        elif idx < self.num_samples and idx + self.batch_size > self.num_samples:
            return [tensor[idx:] for tensor in self.tensors]
        else:
            raise IndexError

    def __add__(self, tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert len(self.tensors) == len(tensors)
        assert all(self_tensor[0].shape == tensor[0].shape for self_tensor, tensor in zip(self.tensors, tensors))

        num_add_samples = tensors[0].size(0)
        self.num_samples = self.num_samples + num_add_samples
        self.tensors = [torch.cat((self_tensor, tensor)) for self_tensor, tensor in zip(self.tensors, tensors)]

    def shuffle(self):
        idx = torch.randperm(self.num_samples, dtype=torch.int64)
        self.tensors = [tensor[idx] for tensor in self.tensors]
        
        


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

    def __init__(self, batchdataset, shuffle=False,
                 pin_memory=False, drop_last=False, device='cuda'):

        self.batch_size = batchdataset.batch_size
        self.dataset = batchdataset
        self.shuffle = shuffle
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

        if loader.shuffle:
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
        batch = self.batchdataset[self.idx]
        # Note Pinning memory was ~10% _slower_ for the test examples I explored
        if self.pin_memory:
            batch = _utils.pin_memory.pin_memory_batch(batch)
        self.idx = self.idx+1
        # move the batch data to device 
        batch = to_device(batch, self.device)
        # return in the form of : xb,yb = (x_cat, x_cont), y
        return (batch[0],batch[1]), batch[2]

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

