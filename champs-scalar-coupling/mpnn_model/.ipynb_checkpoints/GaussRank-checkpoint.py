import numpy as np
from scipy.special import erfinv
from bisect import bisect_left
import pandas as pd 
class GaussRankMap():

    def __init__(self, training_maps=[], coupling_order=[]):
        self.epsilon = 0.001
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower
        
        self.training_maps = training_maps
        self.coupling_order = coupling_order

    def fit_training(self, df, reset=False):
        if self.training_maps and reset == True:
            self.training_maps = []
            self.coupling_order = []
        elif self.training_maps:
            print('GaussRank Mapping already exists.  To overide set reset=True.')
            return
        
        tf = None
        
        for coupling_type in df['type'].unique():
            self.coupling_order.append(coupling_type)
            X = df[df['type']==coupling_type]['scalar_coupling_constant']
            i = np.argsort(X, axis=0)
            j = np.argsort(i, axis=0)

            assert (j.min() == 0).all()
            assert (j.max() == len(j) - 1).all()

            j_range = len(j) - 1
            self.divider = j_range / self.range

            transformed = j / self.divider
            transformed = transformed - self.upper
            transformed = erfinv(transformed)
            
            #print(coupling_type, len(X), len(transformed))
            
            if tf is None:
                tf = transformed.copy(deep=True)

            else:
                tf = tf.append(transformed.copy(deep=True))

            
            training_map = pd.concat([X, transformed], axis=1)
            training_map.columns=['sc','sct']
            training_map.sort_values(['sc'], ascending=[1], inplace=True)
            training_map.reset_index(inplace=True, drop=True)
            
            self.training_maps.append(training_map)
        return tf

    def convert_df(self, df, from_coupling=True):
        #coupling_idx = self.coupling_order.index(coupling_type)
        if from_coupling==True:
            column = 'sc'
            target = 'sct'
            df_column = 'scalar_coupling_constant'
            
        else:
            column = 'sct'
            target = 'sc'
            df_column = 'prediction'

        output = None
        # Do all of the sorts per coupling type in a single operation
        for coupling_type in df['type'].unique():
            training_map = self.training_maps[self.coupling_order.index(coupling_type)]     
            #training_map = cudf.DataFrame.from_pandas(self.training_maps[self.coupling_order.index(coupling_type)])    
            pos = training_map[column].searchsorted(df[df['type']==coupling_type][df_column], side='left')
            pos[pos>=len(training_map)] = len(training_map)-1
            pos[pos-1<=0] = 0
            
            x1 = training_map[column].iloc[pos].reset_index(drop=True)
            x2 = training_map[column].iloc[pos-1].reset_index(drop=True) # larger of the two
            y1 = training_map[target].iloc[pos].reset_index(drop=True)
            y2 = training_map[target].iloc[pos-1].reset_index(drop=True)
            z = df[df['type']==coupling_type].reset_index(drop=False)[['index',df_column]]

            relative = z['index'],(z[df_column]-x2)  / (x1-x2)
            if output is None:
                output = pd.DataFrame(list(zip(relative[0],((1-relative[1])*y2 + (relative[1]*y1)))))
            else:
                output = output.append(pd.DataFrame(list(zip(relative[0],((1-relative[1])*y2 + (relative[1]*y1))))))
           
        output.columns = ['index',target]
        output = output.set_index('index', drop=True)
        # output = output.sort_index()
        # < min or > max
        return output #pd.DataFrame(list(zip(relative[0],((1-relative[1])*y2 + (relative[1]*y1)))))