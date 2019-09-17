# Define dataset 
# coupling_cols = ['atom_index_0', 'atom_index_1','coupling_type','scalar_coupling',
#                  'gaussrank_coupling','fc','sd','pso','dso','id', 'path_index_0', 'path_index_1',
#                  'path_index_2', 'path_index_3', 'path_btype_0', 'path_btype_1',
#                  'path_btype_2', 'path_a_num_0', 'path_a_num_1', 'path_a_num_2']
#
# edge_cols :  ['atom_index_0', 'atom_index_1', 'edge_type', 'distance', 'angle' ]
#
# nodes cols : ['symbol','acceptor', 'donor', 'aromatic',  'hybridization', 'num_h', 'atomic']  
#
###################################################

__all__ = ['run']

#############################################################################################################
#                                                                                                           #
#                                                 Train run                                                 #
#                                                                                                           #
#############################################################################################################

def run(yaml_filepath, fold, type_, freeze_cycle=4, unfreeze_cycle=40):

    
    cfg = load_cfg(yaml_filepath)
    COUPLING_MAX = COUPLING_MAX_DICT[type_]
    
    pretrain_model = model_dict[type_]
    
    ###########################------------- Set Train flags ---------------################################
    num_output = cfg['model']['regression']['num_output']
    OUT_DIR = cfg['dataset']['output_path']
    if num_output == 1:
        out_dir = OUT_DIR + '/submit/scalar_output/'
        # init preditions arrays 
        pred_cv = np.zeros( cfg['train']['train_shape'])
        pred_sub = np.zeros(cfg['train']['test_shape'])
    
    elif num_output == 5:
        out_dir = OUT_DIR + '/submit/multi_output/'
        pred_cv = np.zeros((cfg['train']['train_shape'], 5))
        pred_sub = np.zeros((cfg['train']['test_shape'], 5))
        
    DATA_DIR = cfg['dataset']['input_path']
    normalize = cfg['dataset']['normalize']
    gaussrank=  cfg['dataset']['gaussrank']
    model_name = cfg['train']['model_name']  
    model_name = model_name+ '_fold_%s' %fold 
    batch_size = cfg['train']['batch_size']
    predict_type = cfg['train']['predict_type']
    loss_name = cfg['train']['loss_name']
    predict_type = cfg['model']['regression']['predict_type']
    epochs = cfg['train']['epochs']
    max_lr = cfg['train']['max_lr']
    device = cfg['train']['device']

    
    ############################------------- Init Log file ---------------################################
    log = Logger()
    
        
    log.open(out_dir+'/train/log.train.%s.%s.txt' % (cfg['train']['model_name'], fold), mode='a')
    
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\tconfig file  = %s\n ' % yaml_filepath)
    log.write('\n')
    
    log.write('\nTrain model for type %s and fold %s' %(type_, fold))
    

    ############################------------- Load Datasets ---------------################################
    test= pd.read_csv(DATA_DIR+'/csv/test.csv')
    id_test = test.id.values
    mol_test = test.molecule_name.values

    print('\n Load Train/Validation features for fold %s' %fold)
    validation = gd.read_parquet(DATA_DIR +'/rnn_parquet/fold_%s/%s/validation.parquet'%(fold, type_))
    train = gd.read_parquet(DATA_DIR +'/rnn_parquet/fold_%s/%s/train.parquet' %(fold, type_))

    print('\n Get In-memory Tensor ')

    # Convert train to tensors 
    num_nodes_tensor = from_dlpack(train['num_nodes'].to_dlpack()).long()
    num_edges_tensor = from_dlpack(train['num_edge'].to_dlpack()).long()
    num_coupling_tensor = from_dlpack(train['num_coupling'].to_dlpack()).long()

    node_cols = [i for i in train.columns if re.compile("^node_[0-9]+").findall(i)]
    nodes_matrix = from_dlpack(train[node_cols].to_dlpack()).type(torch.float32)

    edge_cols = [i for i in train.columns if re.compile("^edge_[0-9]+").findall(i)]
    edges_matrix = from_dlpack(train[edge_cols].to_dlpack()).type(torch.float32)

    coupling_cols = [i for i in train.columns if re.compile("^coupling_[0-9]+").findall(i)]
    coupling_matrix = from_dlpack(train[coupling_cols].to_dlpack()).type(torch.float32)

    mol_train = train.molecule_name.unique().to_pandas().values
    train_dataset = TensorBatchDataset(mol_train, 
                                       tensors=[nodes_matrix, edges_matrix, coupling_matrix,
                                                num_nodes_tensor, num_edges_tensor, num_coupling_tensor], 
                                        batch_size=batch_size,
                                       collate_fn=tensor_collate_rnn,
                                       COUPLING_MAX=COUPLING_MAX,
                                        mode='train',
                                        csv='train')
    # convert validation to tensors 
    num_nodes_tensor = from_dlpack(validation['num_nodes'].to_dlpack()).long()
    num_edges_tensor = from_dlpack(validation['num_edge'].to_dlpack()).long()
    num_coupling_tensor = from_dlpack(validation['num_coupling'].to_dlpack()).long()

    node_cols = [i for i in validation.columns if re.compile("^node_[0-9]+").findall(i)]
    nodes_matrix = from_dlpack(validation[node_cols].to_dlpack()).type(torch.float32)

    edge_cols = [i for i in validation.columns if re.compile("^edge_[0-9]+").findall(i)]
    edges_matrix = from_dlpack(validation[edge_cols].to_dlpack()).type(torch.float32)

    coupling_cols = [i for i in validation.columns if re.compile("^coupling_[0-9]+").findall(i)]
    coupling_matrix = from_dlpack(validation[coupling_cols].to_dlpack()).type(torch.float32)


    mol_valid = validation.molecule_name.unique().to_pandas().values
    valid_dataset = TensorBatchDataset(mol_valid, 
                                       tensors=[nodes_matrix, edges_matrix, coupling_matrix,
                                                num_nodes_tensor, num_edges_tensor, num_coupling_tensor], 
                                        batch_size=batch_size,
                                       collate_fn=tensor_collate_rnn,
                                       COUPLING_MAX=COUPLING_MAX,
                                        mode='train',
                                        csv='train')

    del train 
    del validation 

    data = BatchDataBunch.create(train_dataset, valid_dataset, device=device, bs=batch_size)
    
    ############################------------- Load model ---------------################################
    if not gaussrank: 
        net = torch.load('pre_trained_models/coupling_%s_%s_fold_%s_wo_gaussrank.pth'%(type_, pretrain_model, fold))
    else: 
        net = torch.load('pre_trained_models/coupling_%s_%s_fold_%s_gaussrank.pth'%(type_, pretrain_model, fold))
        
    # load grm : 
    data_dir = DATA_DIR + '/rnn_parquet'
    file = glob.glob(data_dir+'/fold_%s/'%fold+'%s/*.csv'%type_)[0]     
    coupling_order = [type_]
    mapping_frames = [pd.read_csv(file)]  
    grm = GaussRankMap(mapping_frames, coupling_order)
    
    
    ############################------------- Fine tune training ---------------################################
    optal = partial(RAdam)
    learn =  Learner(data,
                     net,
                     metrics=None,
                     opt_func=optal,
                     callback_fns=partial(LMAE,
                                        grm=grm,
                                        predict_type=predict_type,
                                        normalize_coupling=normalize, 
                                        coupling_rank=gaussrank))
    
    learn.loss_func = lmae_criterion
    
    learn.split([[learn.model.preprocess,learn.model.message_function, learn.model.update_function, learn.model.readout],
                 [learn.model.rnn_attention],[learn.model.dense_layer, learn.model.predict]])
    
    learn.lr_range(slice(1e-3))

    learn.freeze()
    learn.fit_one_cycle(freeze_cycle, callbacks=[SaveModelCallback(learn,
                                                     every='improvement',
                                                     monitor='LMAE', 
                                                     name=cfg['train']['model_name']+'_fold_%s_frozen_type_%s_'%(fold, type_),
                                                     mode='min')])
    
    learn.unfreeze()
    learn.fit_one_cycle(unfreeze_cycle, max_lr=max_lr, callbacks=[SaveModelCallback(learn,
                                                     every='improvement',
                                                     monitor='LMAE', 
                                                     name=cfg['train']['model_name']+'_fold_%s_pretrained_%s_'%(fold, type_),
                                                     mode='min')])

   
    ############################------------- Build predictions ---------------################################
    valid_dataset = TensorBatchDataset(mol_valid, 
                                    tensors=[nodes_matrix, edges_matrix, coupling_matrix,
                                            num_nodes_tensor, num_edges_tensor, num_coupling_tensor], 
                                    batch_size=batch_size,
                                    collate_fn=tensor_collate_rnn,
                                    COUPLING_MAX=COUPLING_MAX,
                                    mode='test',
                                    csv='train')

    valid_loader = BatchDataLoader(valid_dataset, 
                                   shuffle=False, 
                                   pin_memory=False, 
                                   drop_last=False, 
                                   device='cuda')
    
    log.write('\n Compute predictions for validation data at fold %s\n' %fold)  
    valid_loss, reverse_frame, contributions, molecule_representation = do_test(learn.model,
                                                                           valid_loader,
                                                                           1,
                                                                           1,
                                                                           predict_type,
                                                                           grm,
                                                                           normalize=normalize,
                                                                           gaussrank=gaussrank)
    

    val_loss = valid_loss[-3]
    
    log.write('\nValidation loss is : %s' %val_loss)
    
    log.write('\nSave model to disk')
    torch.save(learn.model, 'models/' + cfg['train']['model_name'] + '_fold_%s_final_save.pth'%fold)
    
    log.write('load test data')
    torch.cuda.empty_cache()
    test = gd.read_parquet(DATA_DIR +'/rnn_parquet/test_%s.parquet'%type_)
    num_nodes_tensor = from_dlpack(test['num_nodes'].to_dlpack())
    num_edges_tensor = from_dlpack(test['num_edge'].to_dlpack())
    num_coupling_tensor = from_dlpack(test['num_coupling'].to_dlpack())
    node_cols = [i for i in test.columns if re.compile("^node_[0-9]+").findall(i)]
    nodes_matrix = from_dlpack(test[node_cols].to_dlpack())
    nodes_matrix = from_dlpack(test[node_cols].to_dlpack()).type(torch.float32)
    edge_cols = [i for i in test.columns if re.compile("^edge_[0-9]+").findall(i)]
    edges_matrix = from_dlpack(test[edge_cols].to_dlpack()).type(torch.float32)
    coupling_cols = [i for i in test.columns if re.compile("^coupling_[0-9]+").findall(i)]
    coupling_matrix = from_dlpack(test[coupling_cols].to_dlpack()).type(torch.float32)

    mol_test  = test.molecule_name.unique().to_pandas().values
    #batch_node, batch_edge, batch_coupling, batch_graussrank, batch_num_node, batch_num_edge, batch_num_coupling
    del test
    
    test_dataset = TensorBatchDataset(mol_test, 
                                    tensors=[nodes_matrix, edges_matrix, coupling_matrix,
                                             num_nodes_tensor, num_edges_tensor, num_coupling_tensor], 
                                    batch_size=batch_size,
                                    collate_fn=tensor_collate_rnn,
                                    COUPLING_MAX=COUPLING_MAX,
                                    mode='test',
                                    csv='test')

    test_loader = BatchDataLoader(test_dataset, 
                                   shuffle=False, 
                                   pin_memory=False, 
                                   drop_last=False, 
                                   device='cuda')
    
    log.write('\n Compute predictions for test data at fold %s\n' %fold)
    test_loss, preds_fold_test, contributions, molecule_representation = do_test(learn.model,
                                                                           valid_loader,
                                                                           1,
                                                                           1,
                                                                           predict_type,
                                                                           grm,
                                                                           normalize=normalize,
                                                                           gaussrank=gaussrank)
    
    # save test predictions 
    log.write('\n Save predictions to disk')
    
    log.write('\n Save Validation frame' )
    clock = "{}".format(datetime.now()).replace(' ','-').replace(':','-').split('.')[0]
    output_name = out_dir + '/cv_%s_%s_%.4f_type_%s_fold_%s.csv.gz'%(clock, pretrain_model, val_loss, type_, fold)
    reverse_frame.to_csv(output_name, index=False,compression='gzip')
    
    # save test predictions 
    log.write('\n Save Test frame' )
    clock = "{}".format(datetime.now()).replace(' ','-').replace(':','-').split('.')[0]
    output_name = out_dir + '/sub_%s_%s_%.4f_type_%s_fold_%s.csv.gz'%(clock, pretrain_model, val_loss, type_, fold)
    preds_fold_test.to_csv(output_name, index=False,compression='gzip')

    net=None
    torch.cuda.empty_cache()
    
    print('\nsuccess!')

def get_parser():
    """Get parser object."""
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)
    
    
    
    parser.add_argument('--fold',  type=int, help='fold id for cv training', required=True)
        
    parser.add_argument('--GPU_id',  type=int, help='gpu to use for training', required=True)
    
    # parser.add_argument('--best_pred_file',  type=str, help='path to best prediction file', required=False)
    
    parser.add_argument('--type',  type=str, help='coupling type', required=False)
    
    parser.add_argument('--freeze_cycle', type=int, help='Number of iterations with frozen weights', required=False)
    
    parser.add_argument('--unfreeze_cycle', type=int, help='Number of iterations with unfrozen weights', required=False)
    
    
    return parser

#############################################################################################################
#                                                                                                           #
#                                               Main function                                               #
#                                                                                                           #
#############################################################################################################
if __name__ == '__main__':
    
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    args = get_parser().parse_args()
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_id)
    
    import sys

    from os.path import dirname, join, abspath
    sys.path.insert(0, abspath(join(dirname(__file__), '..')))

    import cudf as gd
    from fastai.basic_train import *
    from fastai.callbacks import SaveModelCallback
    from functools import partial
    from torch.utils.dlpack import from_dlpack
    
    import glob 
    import warnings
    
    from mpnn_model.build_predictions import do_test 
    from mpnn_model.callback import get_reverse_frame, lmae, LMAE
    from mpnn_model.common import * 
    from mpnn_model.common_constants import * 
    from mpnn_model.dataset import TensorBatchDataset, BatchDataBunch, BatchDataLoader
    from mpnn_model.data_collate import tensor_collate_rnn
    from mpnn_model.GaussRank import GaussRankMap
    from mpnn_model.helpers import load_cfg
    from mpnn_model.model import Net 
    from mpnn_model.radam import * 
    from mpnn_model.train_loss import train_criterion, lmae_criterion
    
    print( '%s: calling main function ... ' % os.path.basename(__file__))    
    
    #cfg, fold, type_, pretrain_model, freeze_cycle=4, unfreeze_cycle=40
    run(args.filename, args.fold, args.type, args.freeze_cycle, args.unfreeze_cycle)

    print('\nsuccess!')