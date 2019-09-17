import torch 


### Helpers for normalization 
NUM_COUPLING_TYPE=8
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
COUPLING_TYPE      = [ COUPLING_TYPE_STATS[i*5  ] for i in range(NUM_COUPLING_TYPE)]
REVERSE_COUPLING_TYPE = dict(zip(range(8), COUPLING_TYPE))

COUPLING_TYPE_MEAN = torch.tensor([COUPLING_TYPE_STATS[i*5+1] for i in range(NUM_COUPLING_TYPE)], dtype=torch.float32).cuda()
COUPLING_TYPE_STD  =  torch.tensor([ COUPLING_TYPE_STATS[i*5+2] for i in range(NUM_COUPLING_TYPE)], dtype=torch.float32).cuda()

COUPLING_MIN_ = [ COUPLING_TYPE_STATS[i*5+3  ] for i in range(NUM_COUPLING_TYPE)]
COUPLING_MAX_ = [ COUPLING_TYPE_STATS[i*5+4  ] for i in range(NUM_COUPLING_TYPE)]

NODE_MAX, EDGE_MAX = 32, 816

COUPLING_MAX_DICT = {'1JHC': 20, '2JHC': 36, '3JHC': 66, '1JHN': 8, '2JHN': 12, '3JHN': 18, '3JHH': 36, '2JHH': 19 }

#--- Set of Categorical modalities 
SYMBOL = ['H', 'C', 'N', 'O', 'F']

# model criterion 
model_dict = { '1JHC': 'lmae', '2JHC': 'lmae', '3JHC': 'lmae', '3JHH': 'lmae',
             '1JHN': 'mlmae' , '2JHN':'mlmae' , '3JHN':'mlmae', '2JHH':'mlmae'}
