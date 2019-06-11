### Pytorch Batch Dataloader
## 🚀 Feature (Submitted to Pytorch as a PR)
A dataloader and dataset that operate at the batch level, rather than the item level, pulling batches from contiguous blocks of memory and avoiding random access patterns in the dataloader.

## Motivation
Loading data item by item and coallating into a batch is very inefficient, particularly in the case of tabular or text data where the items are small.  This is compounded further when you want to use large batch sizes.  By pre shuffling the data each epoch (when required) we can grab each batch as a single read from contiguous memory.  This much faster and scales better with batch size, removing the necessity of multiprocessing, which adds complexity in the form of bus errors, CUDA init issues when forking, etc.  This forking issue was one of my original motivations as it solves the issue of using the dataloader in conjunction with RAPIDS or any other code that calls CUDA before the dataloader workers are forked.  It should also solve the issue on windows with the speed of dataloaders, at least for tabular and text data,  (https://github.com/pytorch/pytorch/issues/12831) as spawning is not necessary.  

Using the proposed method results in better GPU utilization, and better throughput when training in the tests on tabular data that I've run.  With no multiprocessing I've measured a 5-15% improvement* in throughput over an 8 worker vanilla dataloader (more were tried but it maxed out at 8).  I've also been able to increase batch sizes for tabular data into the 800K+ range with no loss of accuracy and get a 2x performance improvement over the best multiprocessor dataloader I could run without running into bus error issues that cropped up with large batch sizes.

*depends on tensor and batch size

## Pitch

I've created source for a batch dataloader and batch dataset modelled after their vanilla counterparts and would love to see it integrated into the PyTorch repo.  Usage is similar, and I've tried to stick to the pytorch variable naming and formatting.

The code should be ready to go; I've tested it with both base pytorch and with ignite, but more eyes on it would definitely be beneficial, particularly in use cases beyond tabular like text or small images.  It should be applicable to anyone who isn't doing large images or a lot of image augmentation.  It's undergone an internal (NVidia) review of @ptrblck who was immensely helpful in refining it and @ngimel who reviewed the codebase and had helpful suggestions regarding memory pinning.  

I'm happy to work with the team to create test cases similar to those for dataset and dataloader and would love feedback on it.

## Alternatives

One possible solution to the CUDA Init before fork issue is to spawn, however as seen in windows this is significantly slower and I had trouble getting it working.   

## Additional context

We're also working on versions of this that work with larger than CPU memory datasets and on a version that works in GPU memory doing a 0-copy transform of a rapids cudf dataframe via dlpack.


