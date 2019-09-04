### RAPIDS.AI Deep Learning Repo
This repository is the home of our efforts to integrate RAPIDS acceleration of dataframes on GPU into popular deep learning frameworks.  The work can be broken down into three main sections:

 - Dataloaders and preprocessing functionality developed to help provide connectivity between RAPIDS cuDF dataframes and the different deep learning libraries available.  
 - Improvements to optimizers through the fusion of GPU operations.
 - Examples of the use of each of the above in competitions or on real world datasets.

Each deep learning library is contained within it's own subfolder, with the different dataloader options and examples contained within further subfolders.  For now our focus is on PyTorch, however we expect to add other libraries in the future.
