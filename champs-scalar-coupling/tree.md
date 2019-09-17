.
├── build_data
│   ├── __init__.py
│   ├── lib
│   │   ├── include.py
│   │   ├── __init__.py
│   │   ├── net
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__
│   │   │   └── rate.py
│   │   └── utility
│   │       ├── draw.py
│   │       ├── file.py
│   │       ├── __init__.py
│   ├── common.py
│   ├── data-cudf.py
│   ├── data.py
│   ├── parallel_process.py
│   ├── atom_features.py
│   ├── baseline_node_frame_from_csv.ipynb
│   ├── build_baseline_dataframes.ipynb
│   ├── build_preds_from_checkpoints.ipynb
│   ├── build_stack_train_validation.ipynb
│   ├── build_train_validation.ipynb
│   ├── build_train_validation_rnn.ipynb
│   ├── build_train_validation_rnn_per_type.ipynb
│   ├── build_train_validation_rnn_scalar.ipynb
├── experiments
│   ├── \*.yaml
├── merge_predictions_per_type.ipynb
├── models
│   ├── \*.pth
├── mpnn_model
│   ├── __init__.py
│   ├── lib
│   │   ├── include.py
│   │   ├── __init__.py
│   │   ├── net
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__
│   │   │   └── rate.py
│   │   └── utility
│   │       ├── draw.py
│   │       ├── file.py
│   │       ├── __init__.py
│   ├── build_predictions.py
│   ├── callback.py
│   ├── common_constants.py
│   ├── common_model.py
│   ├── common.py
│   ├── data_collate.py
│   ├── data.py
│   ├── dataset.py
│   ├── GaussRank.py
│   ├── helpers.py
│   ├── message_passing.py
│   ├── model.py
│   ├── parallel_process.py
│   ├── radam.py
│   ├── regression_head.py
│   ├── RNN_attention.py
│   └── train_loss.py
├── pre_trained_models
│   ├── \*.pth
├── scripts
│   ├── bootsrap_train_mpnn_rnn.py
│   ├── train_mpnn.py
│   ├── train_mpnn_rnn.py
│   └── train_type.py
├── train_MPNN_RNN.ipynb
├── train_MPNN_RNN_SINGLE_TYPE.ipynb
├── save_pretrained_single_models.ipynb
├── README.md
└── tree.md