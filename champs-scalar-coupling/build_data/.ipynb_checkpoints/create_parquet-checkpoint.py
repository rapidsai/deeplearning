#
#
#
#
# This module aims to create a parquet file from .pkl graph files
#
# It also gives the possibility to compute the gaussrank and create 
# train/validation .parquet  files for each fold with. each line represents
# a molecules and its information in the following order : 
#     'molecule_name',
#     'num_node', 'num_edge','num_coupling',
#     'node_dim', 'edge_dim','coupling_dim', 
#     'node_0', ...., 'node_NODE_MAX*7'
#     'edge_0', ...., 'edge_EDGE_MAX*5'
#     'coupling_0', ...., 'coupling_COUPLING_MAX*9'
#     'gaussrank_0', ....., 'gaussrank_COUPLING_MAX'
#
#
#
#####################################################################################


import glob
from parallel_process import parallel_process
from data import *
from common import *
from GaussRank import *

from functools import partial
import pandas as pd
from time import time 

NODE_MAX = 32
EDGE_MAX = 816
COUPLING_MAX = 136


def get_one_vector_from_graph(molecule_file): 
    '''
      - molecule file:  path to %molecule_name.pickle
    Returns: 
      Convert the pickled graph to a padded vector with all the molecule information 
    '''
    molecule_name = molecule_file.split('/')[-1].strip('.pickle')
    graph = read_pickle_from_file(molecule_file)
    molecule_name = graph.molecule_name
    node_feats = np.concatenate(graph.node,-1)
    edge_feats = np.concatenate(graph.edge,-1)
    edge_feats = np.concatenate([graph.edge_index, edge_feats], -1)
    
    coupling = np.concatenate([graph.coupling.index, graph.coupling.type.reshape(-1, 1), 
                               graph.coupling.value.reshape(-1,1), graph.coupling.contribution,
                               graph.coupling.id.reshape(-1,1)], -1)

    num_node, node_dim = node_feats.shape 
    num_edge, edge_dim = edge_feats.shape 
    num_coupling, coupling_dim = coupling.shape
    
    infor = [molecule_name, num_node, num_edge, num_coupling, node_dim, edge_dim, coupling_dim]
    return  infor, node_feats.reshape(num_node*node_dim), edge_feats.reshape(num_edge*edge_dim), coupling.reshape(num_coupling*coupling_dim)

def build_general_frame(graph_dir, parquet_dir='/rapids/notebooks/srabhi/champs-2019/input/parquet/'):
    ''' 
        Args: 
            - graph_dir to use for getting molecule information: 
               - graph1: one_hot encoding for categorical values + acutal value of scalar coupling constant
               - graph2: label encoding for cats + actual scalar coupling
               - graph3: one_hot encoding for cats + normalized scalar coupling
               - graph4: label encoding for cats + normalized scalar coupling
            - parquet_dir: 
               - output directory where to store the general parquet frame 
    '''

    files = glob.glob(graph_dir+'/*.pickle')
    tabular_data = parallel_process(files, get_one_vector_from_graph)

    nodes = []
    infos = []
    edges = []
    coupling = []
    for i in tabular_data:
        infos.append(i[0])
        nodes.append(i[1])
        edges.append(i[2])
        coupling.append(i[3])
    info_frame, node_frame, edge_frame, coupling_frame = (pd.DataFrame(infos,columns=['molecule_name', 'num_node', 'num_edge', 
                                                                                  'num_coupling', 'node_dim', 'edge_dim', 'coupling_dim']),
                                                        pd.DataFrame(nodes), pd.DataFrame(edges), pd.DataFrame(coupling))

    ### Get a multiple 8 for gpu ops 
    # pad  29 nodes to node_max 32 : 
    pad_cols = 21
    d = dict.fromkeys([str(i) for i in range(node_frame.shape[1], node_frame.shape[1]+pad_cols)], 0.0)
    node_frame = node_frame.assign(**d).fillna(0.0)

    # pad edge_max 812 to 816
    pad_cols = 20
    d = dict.fromkeys([str(i) for i in range(edge_frame.shape[1], edge_frame.shape[1]+pad_cols)], 0.0)
    edge_frame = edge_frame.assign(**d).fillna(0.0)

    # pad coupling_max to 136
    pad_cols = 9
    d = dict.fromkeys([str(i) for i in range(coupling_frame.shape[1], coupling_frame.shape[1]+pad_cols)], 0.0)
    coupling_frame = coupling_frame.assign(**d).fillna(0.0)

    # concat the whole frame 
    general_frame = pd.concat([info_frame, node_frame, edge_frame, coupling_frame], axis=1)
    general_frame = general_frame.fillna(0.0)

    print('Dataframe created for %s molecules' %general_frame.shape[0])
    cols = ['molecule_name', 'num_node', 'num_edge', 'num_coupling', 'node_dim', 'edge_dim', 'coupling_dim'] + \
    ['node_%s'%i for i in range(NODE_MAX*7)] + ['edge_%s'%i for i in range(EDGE_MAX*5)] + ['coupling_%s'%i for i in range(COUPLING_MAX*9)]
    general_frame.columns = cols
    general_frame.to_parquet(os.path.join(parquet_dir, 'general_frame.parquet'))

    
def build_test_data(data, DATA_DIR = '/rapids/notebooks/srabhi/champs-2019/input'):
    #data = gd.read_parquet(DATA_DIR +'/parquet/general_frame.parquet')
    csv = 'test'
    df = pd.read_csv(DATA_DIR + '/csv/%s.csv'%csv)
    id_test = gd.DataFrame()
    mol_test = df.molecule_name.unique()
    id_test['molecule_name'] = mol_test
    test_data = id_test.merge(data, on='molecule_name', how='left')
    tmp = pd.DataFrame(np.zeros((45772, 136),  dtype=float))
    tmp.columns = ['gaussrank_%s'%i for i in range(136)]
    tmp = gd.from_pandas(tmp)
    tmp['molecule_name'] = test_data.molecule_name
    test = tmp.merge(test_data, on='molecule_name', how='left')
    test.to_parquet(DATA_DIR +'/parquet/test_frame.parquet')

    
def build_cv_ranks_parquet(data, fold, DATA_DIR = '/rapids/notebooks/srabhi/champs-2019/input'):
    print(fold)
    ### Get data 
    split_train = 'train_split_by_mol_hash.%s.npy'%fold
    split_valid = 'valid_split_by_mol_hash.%s.npy'%fold
    id_train_ = np.load(DATA_DIR + '/split/%s'%split_train,allow_pickle=True)
    id_valid_ = np.load(DATA_DIR + '/split/%s'%split_valid,allow_pickle=True)
    df = pd.read_csv(DATA_DIR + '/csv/train.csv')
    #data = gd.read_parquet(DATA_DIR+'/parquet/general_frame.parquet')

    train = df[df.molecule_name.isin(id_train_)]
    validation = df[df.molecule_name.isin(id_valid_)]

    # Get GaussRank of coupling values 
    t0 = time()
    grm = GaussRankMap()
    transformed_training = grm.fit_training(train, reset=True)
    transformed_validation = grm.convert_df(validation, from_coupling=True)
    validation['transformed_coupling'] =  transformed_validation
    train['transformed_coupling'] = transformed_training
    print('Getting gaussrank transformation for train/validation data took %s seconds' %(time()-t0))
    print(grm.coupling_order)
    # Get the rank coupling values at the molecule level and pad coupling rank values to 136 : 
    validation_gaussrank = validation.groupby('molecule_name').apply(lambda x : x['transformed_coupling'].values)
    train_gaussrank = train.groupby('molecule_name').apply(lambda x : x['transformed_coupling'].values)
    
    val_ranks = pd.DataFrame(validation_gaussrank.tolist()).fillna(0.0)
    num_cols = val_ranks.shape[1]
    pad_cols = 136 - num_cols
    d = dict.fromkeys([str(i) for i in range(num_cols, num_cols+pad_cols)], 0.0)
    val_ranks = val_ranks.assign(**d)
    val_ranks = val_ranks.astype(float)
    val_ranks.columns = ['gaussrank_%s'%i for i in range(136)]
    val_ranks['molecule_name'] = validation_gaussrank.index

    train_ranks = pd.DataFrame(train_gaussrank.tolist()).fillna(0.0)
    num_cols = train_ranks.shape[1]
    pad_cols = 136 - num_cols
    d = dict.fromkeys([str(i) for i in range(num_cols, num_cols+pad_cols)], 0.0)
    train_ranks = train_ranks.assign(**d)
    train_ranks = train_ranks.astype(float)
    train_ranks.columns = ['gaussrank_%s'%i for i in range(136)]
    train_ranks['molecule_name'] = train_gaussrank.index

    # Merge with node /edge/coupling frame 
    id_valid = gd.DataFrame()
    id_valid['molecule_name'] = id_valid_
    valid_data = id_valid.merge(data, on='molecule_name', how='left')

    id_valid = gd.DataFrame()
    id_valid['molecule_name'] = id_valid_
    valid_data = id_valid.merge(data, on='molecule_name', how='left').to_pandas()
    validation_frame = pd.merge(valid_data, val_ranks, on='molecule_name', how='left')


    # Merge with node /edge/coupling frame 
    id_train= gd.DataFrame()
    id_train['molecule_name'] =  id_train_
    train_data = id_valid.merge(data, on='molecule_name', how='left')

    id_train = gd.DataFrame()
    id_train['molecule_name'] =  id_train_
    train_data = id_train.merge(data, on='molecule_name', how='left').to_pandas()   
    training_frame = pd.merge(train_data, train_ranks, on='molecule_name', how='left')
    
    # Save parquet files for fold 
    parquet_dir = DATA_DIR + '/parquet/fold_%s' %fold                      
    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)
    training_frame.to_parquet(parquet_dir+'/train.parquet')
    validation_frame.to_parquet(parquet_dir+'/validation.parquet')                
                           
    # save mapping 
    for i, (type_, frame) in enumerate(zip(grm.coupling_order, grm.training_maps)): 
        frame.to_csv(parquet_dir+'/mapping_type_%s_order_%s.csv'%(type_, i), index=False)
    pass 
