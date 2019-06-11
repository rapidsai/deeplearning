import torch
from torch import _utils

class BatchDataLoader(object):
    """Batch Data loader. Takes in a batch dataset and returns iterators that return whole batches of data.
    Arguments:
        batchdataset (BatchDataset): dataset from which to load the data.
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
    """

    def __init__(self, batchdataset, shuffle=False,
                 pin_memory=False, drop_last=False):
        self.batchdataset = batchdataset
        self.batch_size = batchdataset.batch_size

        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last


    def __iter__(self):
        return _BatchDataLoaderIter(self)

    def __len__(self):
        if self.drop_last and self.batchdataset.num_samples%self.batch_size != 0:
            return len(self.batchdataset)-1
        else:
            return len(self.batchdataset)

    
class _BatchDataLoaderIter(object):
    """Iterates once over the BatchDataLoader's batchdataset, shuffling if requested"""
    def __init__(self, loader):
        self.batchdataset = loader.batchdataset
        self.batch_size = loader.batch_size
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.drop_last = loader.drop_last

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
        return batch

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self
