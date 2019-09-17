#
#
#
#      This module aims to create molecule graphs from Kaggle data and rdkit
#
# It also give the possibilit to create cv folds as .npy files with molecule names 
#
#
#
#####################################################################################
#from atom_features import * 
from collections import defaultdict

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import rdkit.Chem.Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
DrawingOptions.bondLineWidth=1.8

from rdkit.Chem.rdmolops import SanitizeFlags
 
import os 
from functools import partial
import argparse
import pandas as pd 
import cudf as gd
import numpy as np 

import scipy
from sklearn import preprocessing


# __all__ = ['make_graph', 'do_one', 'run_convert_to_graph', 'run_make_split' ]


## Helpers for feature extraction #####################################################

COUPLING_TYPE_STATS=[
    #type   #mean, std, min, max
    '1JHC',  94.9761528641869,   18.27722399839607,   66.6008,   204.8800,
    '2JHC',  -0.2706244378832,    4.52360876732858,  -36.2186,    42.8192,
    '3JHC',   3.6884695895355,    3.07090647005439,  -18.5821,    76.0437,
    '1JHN',  47.4798844844683,   10.92204561670947,   24.3222,    80.4187,
    '2JHN',   3.1247536134185,    3.67345877025737,   -2.6209,    17.7436,
    '3JHN',   0.9907298624944,    1.31538940138001,   -3.1724,    10.9712,
    '2JHH', -10.2866051639817,    3.97960190019757,  -35.1761,    11.8542,
    '3JHH',   4.7710233597359,    3.70498129755812,   -3.0205,    17.4841,
]

NUM_COUPLING_TYPE = len(COUPLING_TYPE_STATS)//5

COUPLING_TYPE_MEAN = [ COUPLING_TYPE_STATS[i*5+1] for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE_STD  = [ COUPLING_TYPE_STATS[i*5+2] for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE      = [ COUPLING_TYPE_STATS[i*5  ] for i in range(NUM_COUPLING_TYPE)]


#--- Set of Categorical modalities 
SYMBOL = ['H', 'C', 'N', 'O', 'F']

BOND_TYPE = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

HYBRIDIZATION=[
    #Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    #Chem.rdchem.HybridizationType.SP3D,
    #Chem.rdchem.HybridizationType.SP3D2,
]

def one_hot_encoding(x, set):
    """
    One-Hot Encode categorical variables 
    """
    one_hot = [int(x == s) for s in set]
    if 0:
        if sum(one_hot)==0: print('one_hot_encoding() return NULL!', x, set)
    return one_hot

def label_encoding(x, set): 
    """
    Encode categorical variables to int Ids 
    """
    try: 
        return set.index(x)+1
    except: 
        return 0


''' Graph Structure 
node_feature :
    category 
        (symbol,SYMBOL)  #5 
        (acceptor,) #1
        (donor,   ) #1
        (aromatic,) #1
        one_hot_encoding(hybridization,HYBRIDIZATION) #3
    real  
        (num_h,  ) #1
        (atomic, ) #1
        
        
edge_feature :
    category 
        (bond_type,BOND_TYPE)  #4 
    real  
        np.digitize(distance,DISTANCE) #1
        angle #1
        
 coupling: Structure
         id: 
         contributions: 
         index: 
         type: 
         value:
'''

#############################################################################################################
#                                                                                                           #
#                                 Molecule graph representation                                             #
#                                                                                                           #
#############################################################################################################

def make_graph(molecule_name, gb_structure, gb_scalar_coupling,
               categorical_encoding='one_hot', normalize_coupling=False, rank=False) :
    """
    make_graph --> returns graph as 'Struct' object  (see /lib/utility/file.py)
    
    Args: 
        - molecule_name :  (str)
        - gb_structure (DataFrame GroupBy):  groupby structure:  data groupped by molecule name 
        - gb_scalar_coupling (DataFrame GroupBy): The coupling contributions data groupped by molecule name 
        - categorical_encoding (str):  How represent categorical variables : label vs one-hot enconding 
        - rank: Transform values into norma distribution 
    """
    #---- Coupling informatiom
    # ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'],
    df = gb_scalar_coupling.get_group(molecule_name)
    
    coupling_index = np.array([ COUPLING_TYPE.index(t) for t in df.type.values ], np.int32)
    scalar_coupling_constant = df.scalar_coupling_constant.values
    
    if normalize_coupling:
        coupling_mean = np.array([COUPLING_TYPE_MEAN[x] for x in coupling_index], np.float32)
        coupling_std = np.array([COUPLING_TYPE_STD[x] for x in coupling_index], np.float32)
        scalar_coupling_constant = (scalar_coupling_constant - coupling_mean) / coupling_std
    if rank: 
        scalar_tranform = df.transform.values
        
    coupling = Struct(
        id = df.id.values,
        contribution = df[['fc', 'sd', 'pso', 'dso']].values,
        index = df[['atom_index_0', 'atom_index_1']].values,
        type = coupling_index,
        value = scalar_coupling_constant,
    )

    #---- Molecule structure information 
    df = gb_structure.get_group(molecule_name)
    df = df.sort_values(['atom_index'], ascending=True)
    # ['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z']
    a   = df.atom.values.tolist()
    xyz = df[['x','y','z']].values
    mol = mol_from_axyz(a, xyz)

    #---
    assert( #check
       a == [ mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    )

    #--- Atoms information 
    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)


    if categorical_encoding =='one_hot':
        ## ** node features **
        num_atom = mol.GetNumAtoms()
        symbol   = np.zeros((num_atom,len(SYMBOL)),np.uint8) #category
        acceptor = np.zeros((num_atom,1),np.uint8) #bool
        donor    = np.zeros((num_atom,1),np.uint8) #bool
        aromatic = np.zeros((num_atom,1),np.uint8) #bool
        hybridization = np.zeros((num_atom,len(HYBRIDIZATION)),np.uint8) #category
        num_h  = np.zeros((num_atom,1),np.float32) #real
        atomic = np.zeros((num_atom,1),np.float32) #real 
        for i in range(num_atom):
            atom = mol.GetAtomWithIdx(i)
            symbol[i]        = one_hot_encoding(atom.GetSymbol(),SYMBOL)
            aromatic[i]      = atom.GetIsAromatic()
            hybridization[i] = one_hot_encoding(atom.GetHybridization(),HYBRIDIZATION)

            num_h[i]  = atom.GetTotalNumHs(includeNeighbors=True)
            atomic[i] = atom.GetAtomicNum()
            
        for t in range(0, len(feature)):
            if feature[t].GetFamily() == 'Donor':
                for i in feature[t].GetAtomIds():
                    donor[i] = 1
            elif feature[t].GetFamily() == 'Acceptor':
                for i in feature[t].GetAtomIds():
                    acceptor[i] = 1
                            
        ## ** edge features **
        num_edge = num_atom*num_atom - num_atom
        edge_index = np.zeros((num_edge,2), np.uint8) # int tuples 
        bond_type  = np.zeros((num_edge,len(BOND_TYPE)), np.uint8) #category
        distance   = np.zeros((num_edge,1),np.float32) #real
        angle      = np.zeros((num_edge,1),np.float32) #real
        relative_angle = np.zeros((num_edge,1),np.float32) #real

        norm_xyz = preprocessing.normalize(xyz, norm='l2')

        ij=0
        for i in range(num_atom):
            for j in range(num_atom):
                if i==j: continue
                edge_index[ij] = [i,j]

                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bond_type[ij] = one_hot_encoding(bond.GetBondType(),BOND_TYPE)

                distance[ij] = ((xyz[i] - xyz[j])**2).sum()**0.5
                angle[ij] = (norm_xyz[i]*norm_xyz[j]).sum()

                ij+=1
                       
    elif categorical_encoding =='label': 
        ## ** node features **
        num_atom = mol.GetNumAtoms()
        symbol   = np.zeros((num_atom,1),np.uint8) #category
        acceptor = np.zeros((num_atom,1),np.uint8) #bool
        donor    = np.zeros((num_atom,1),np.uint8) #bool
        aromatic = np.zeros((num_atom,1),np.uint8) #bool
        hybridization = np.zeros((num_atom,1),np.uint8) #category
        num_h  = np.zeros((num_atom,1),np.float32) #real
        atomic = np.zeros((num_atom,1),np.float32) #real
        
        for i in range(num_atom):
            atom = mol.GetAtomWithIdx(i)
            symbol[i]        = label_encoding(atom.GetSymbol(), SYMBOL)
            aromatic[i]      = atom.GetIsAromatic()
            hybridization[i] = label_encoding(atom.GetHybridization(),HYBRIDIZATION)
            num_h[i]  = atom.GetTotalNumHs(includeNeighbors=True)
            atomic[i] = atom.GetAtomicNum()
                
        for t in range(0, len(feature)):
            if feature[t].GetFamily() == 'Donor':
                for i in feature[t].GetAtomIds():
                    donor[i] = 1
            elif feature[t].GetFamily() == 'Acceptor':
                for i in feature[t].GetAtomIds():
                    acceptor[i] = 1
                    
        ## ** edge features **
        num_edge = num_atom*num_atom - num_atom
        edge_index = np.zeros((num_edge,2), np.uint8) # int tuples 
        bond_type  = np.zeros((num_edge,1), np.uint8) #category
        distance   = np.zeros((num_edge,1),np.float32) #real
        angle      = np.zeros((num_edge,1),np.float32) #real

        norm_xyz = preprocessing.normalize(xyz, norm='l2')

        ij=0
        for i in range(num_atom):
            for j in range(num_atom):
                if i==j: continue
                edge_index[ij] = [i,j]

                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bond_type[ij] = label_encoding(bond.GetBondType(),BOND_TYPE)

                distance[ij] = ((xyz[i] - xyz[j])**2).sum()**0.5
                angle[ij] = (norm_xyz[i]*norm_xyz[j]).sum()
                ij+=1
         
    else : 
            raise Exception(f"""{categorical_encoding} invalid categorical labeling""")
                       
    ##---- Define the graph structure 

    graph = Struct(
        molecule_name = molecule_name,
        smiles = Chem.MolToSmiles(mol),
        axyz = [a,xyz],

        node = [symbol, acceptor, donor, aromatic, hybridization, num_h, atomic,],
        edge = [bond_type, distance, angle],
        edge_index = edge_index,

        coupling = coupling,
    )
    return graph


#############################################################################################################
#                                                                                                           #
#                                            Load Champs Datasets                                           #
#                                                                                                           #
#############################################################################################################
def read_champs_xyz(xyz_file):
    line = read_list_from_file(xyz_file, comment=None)
    num_atom = int(line[0])
    xyz=[]
    symbol=[]
    for n in range(num_atom):
        l = line[1+n]
        l = l.replace('\t', ' ').replace('  ', ' ')
        l = l.split(' ')
        symbol.append(l[0])
        xyz.append([float(l[1]),float(l[2]),float(l[3]),])
    return symbol, xyz


def mol_from_axyz(symbol, xyz):
    charged_fragments = True
    quick   =  True
    charge  = 0
    atom_no = get_atomicNumList(symbol)
    mol     = xyz2mol(atom_no, xyz, charge, charged_fragments, quick)
    return mol

def load_csv():
    """
    load_csv --> load the GroupBy DataFrames (Grouping by molecule names)
    """

    DATA_DIR = '/champs-2019/input'

    #structure
    df_structure = pd.read_csv(DATA_DIR + '/csv/structures.csv')

    #coupling
    df_train = pd.read_csv(DATA_DIR + '/csv/train_transform.csv')
    df_test  = pd.read_csv(DATA_DIR + '/csv/test.csv')
    df_test['scalar_coupling_constant']=0
    df_test['transform']=0
    df_scalar_coupling = pd.concat([df_train,df_test])
    df_scalar_coupling_contribution = pd.read_csv(DATA_DIR + '/csv/scalar_coupling_contributions.csv')
    df_scalar_coupling = pd.merge(df_scalar_coupling, df_scalar_coupling_contribution,
            how='left', on=['molecule_name','atom_index_0','atom_index_1','atom_index_0','type'])

    gb_scalar_coupling = df_scalar_coupling.groupby('molecule_name')
    gb_structure       = df_structure.groupby('molecule_name')
    return gb_structure, gb_scalar_coupling


#############################################################################################################
#                                                                                                           #
#                                            Tests check .                                                  #
#                                                                                                           #
#############################################################################################################
def run_check_xyz():
    ''' check xyz files '''

    xyz_dir = '/champs-2019/input/structures'
    name =[
        'dsgdb9nsd_000001',
        'dsgdb9nsd_000002',
        'dsgdb9nsd_000005',
        'dsgdb9nsd_000007',
        'dsgdb9nsd_037490',
        'dsgdb9nsd_037493',
        'dsgdb9nsd_037494',
    ]
    for n in name:
        xyz_file = xyz_dir + '/%s.xyz'%n

        symbol, xyz = read_champs_xyz(xyz_file)
        mol = mol_from_axyz(symbol, xyz)

        smiles = Chem.MolToSmiles(mol)
        print(n, smiles)

        image = np.array(Chem.Draw.MolToImage(mol,size=(128,128)))
        image_show('',image)
        cv2.waitKey(0)



def run_check_graph():
    ''' check graph construction '''

    gb_structure, gb_scalar_coupling = load_csv()

    molecule_name = 'dsgdb9nsd_000001'
    normalize_coupling = False
    graph = make_graph(molecule_name, gb_structure, gb_scalar_coupling, normalize_coupling)


    print('')
    print(graph)
    print('graph.molecule_name:', graph.molecule_name)
    print('graph.smiles:', graph.smiles)
    print('graph.node:', np.concatenate(graph.node,-1).shape)
    print('graph.edge:', np.concatenate(graph.edge,-1).shape)
    print('graph.edge_index:', graph.edge_index.shape)
    print('-----')
    print('graph.coupling.index:', graph.coupling.index.shape)
    print('graph.coupling.type:', graph.coupling.type.shape)
    print('graph.coupling.value:', graph.coupling.value.shape)
    print('graph.coupling.contribution:', graph.coupling.contribution.shape)
    print('graph.coupling.id:', graph.coupling.id)
    print('')

    exit(0)
    zz=0



#############################################################################################################
#                                                                                                           #
#                                             Build graphs                                                  #
#                                                                                                           #
#############################################################################################################

def do_one(p, categorical_encoding='one_hot', normalize_coupling=False):
    ''' Create and save the graph of molecule name: p '''
    i, molecule_name, gb_structure, gb_scalar_coupling, graph_file = p

    g = make_graph(molecule_name, gb_structure, gb_scalar_coupling, categorical_encoding, normalize_coupling)
    print(i, g.molecule_name, g.smiles)
    write_pickle_to_file(graph_file,g)

##----
def run_convert_to_graph(categorical_encoding='one_hot', normalize_coupling = False , graph_dir='/champs-2019/input/structure/graph1'):
    '''
    Convert Train and Test data to graph structures and save each graph as .pkl file in graph_dir path 
    '''
    # graph_dir = '/champs-2019/input/structure/graph1'
    os.makedirs(graph_dir, exist_ok=True)

    gb_structure, gb_scalar_coupling = load_csv()
    molecule_names = list(gb_scalar_coupling.groups.keys())
    molecule_names = np.sort(molecule_names)

    param=[]
    for i, molecule_name in enumerate(molecule_names):

        graph_file = graph_dir + '/%s.pickle'%molecule_name
        p = (i, molecule_name, gb_structure, gb_scalar_coupling, graph_file)

        if i<2000:
            do_one(p, categorical_encoding, normalize_coupling)
        else:
            param.append(p)

    if 1:
        pool = mp.Pool(processes=16)
        pool.map(partial(do_one, categorical_encoding=categorical_encoding, normalize_coupling=normalize_coupling), param)



#############################################################################################################
#                                                                                                           #
#                                      Build Cross-Validation folds                                         #
#                                                                                                           #
#############################################################################################################

def run_make_split(folds):
    '''
    Methods for building cv folds:  each fold is represented by two .npy files of unique molecule names in train / validation data fold
    
    Arguments : 
        folds (type: int): number of validation folds 
    
    save train / valid npy files with related molecule names.     
    '''

    split_dir = '/champs-2019/input/split'

    csv_file = '/champs-2019/input/csv/train.csv'

    print('Read train data')
    df = gd.read_csv(csv_file)
    df['molecule_name_hash'] = df['molecule_name'].data.hash()
    
    
    # get unique molecules 
    print('Get unique molecules names')
    molecule_names = df['molecule_name'].unique().to_pandas().values
    molecule_names = np.sort(molecule_names)
    
    print('Create train / validation folds')
    debug_split = molecule_names[:1000]
    np.save(split_dir + '/debug_split_by_mol.%d.npy'%len(debug_split), debug_split)
    print(debug_split[0:5])  #'dsgdb9nsd_001679'
    
    for fold in range(folds): 
        print(fold)
        mask = df['molecule_name_hash']%folds==fold
        tr, va = df[~mask]['molecule_name'],df[mask]['molecule_name']
        train_split = tr.unique().to_pandas().values
        valid_split = va.unique().to_pandas().values
        np.save(split_dir + '/train_split_by_mol_hash.%d.npy'%(fold),train_split)
        np.save(split_dir + '/valid_split_by_mol_hash.%d.npy'%(fold),valid_split)

    pass   



#############################################################################################################
#                                                                                                           #
#                                            main program                                                   #
#                                                                                                           #
#############################################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build graph and cross-validation data')
    parser.add_argument('--cv', default=False, action ='store_true', help='whether to build cv npy folds or not')
    parser.add_argument('--folds', type=int, help='number of validation folds')
    parser.add_argument('--categorical_encoding', type=str, help='How to encode categorical values: "one_hot" vs "label"' )
    parser.add_argument('--graph_dir', type=str, help='output dir for saving the graph structure of all the molecules')
    parser.add_argument('--normalize', default=False, action ='store_true', help='whether to normalize couplings')
    parser.add_argument('--ranktransform', default=False, action ='store_true', help='whether to comput the normal dist of coupling')
    
    args = parser.parse_args()
    
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    # test the graph structure : run_check_graph()

    if args.cv:
        # Build cv folds 
        run_make_split(args.folds)
    
    # Convert data to graphs 
    run_convert_to_graph(args.categorical_encoding, args.normalize, args.graph_dir)
