This document explains how to use the jupyter notebooks in this folder.

First, collect the data at the following location:
https://recsys.trivago.cloud/challenge/dataset/
(you need to sign up to get access)

Once you have the data set, you need to create the feature set from the data provided.
Navigate to the FeatureEngineering subfolder, in this subfolder, there are two folders:

rapids and pandas

We have provided you with the same routines using both rapids cuDF and pandas. You can choose which you wish to try. 

Within either secondary folder (rapids or pandas) you will find four available feature engineering workflows.

You must run the create_data_pair_comparison_<type>.ipynb (first)

Currently the rapids version requires cuDF version 0.7 because of a masking bug (https://github.com/rapidsai/cudf/issues/2141). 

After you have successfully completed feature engineering. Verify the exported parquet files exist in the rsc19/cache/. Once verified, you can proceed to the processing. The processing subfolder contains 3 notebooks, representing the three available model types. Choose one and run all cells to see speeds of each model. 
