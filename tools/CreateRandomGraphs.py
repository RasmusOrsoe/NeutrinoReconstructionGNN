# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import time
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from copy  import deepcopy
from random import shuffle
scalar =  pd.read_csv('C:\\applied_ML\\final_project\\data\\scalar.csv')
scalar = scalar.loc[:,scalar.columns[1:len(scalar.columns)]]


## BELOW 30 .pkl FILES ARE CREATED, EACH CONTAINING 10.000 GRAPHS IN THE RANDOM EDGE CONFIGURATION.
## THE FILES ARE CREATED BY LOADING THE PADDED .csv EVENT FILES CREATED EARLIER.
## NOTE: HERE THE NODE FEATURES ARE SCALED 
for j in range(0,30):
    graphs = list()
    start = time.time()
    count = 1
    events  = pd.Series.reset_index(scalar['event_no'][0 + j*10000: 10000 + j*10000], drop = True)
    for h in range(0,len(events)):
        start_g = time.time()
        event = pd.read_csv('E:\\final_project\\data\\events_padded\\%s.csv'%events[h])
        upper = np.repeat(event.index.values.tolist(),4).tolist()
        lower = deepcopy(upper)
        shuffle(lower)
        edge_index = torch.tensor([upper,
                                           lower], dtype = torch.long)
            
        if j == 0:
            scaler = StandardScaler()
            scaler.fit(event.loc[:,event.columns[0:5]])
        x = torch.tensor(scaler.transform(event.loc[:,event.columns[0:5]]).tolist(),dtype = torch.float)
        y = torch.tensor(scalar.loc[h,['true_primary_energy','true_primary_time',
                                                                            'true_primary_position_x','true_primary_position_y',
                                                                            'true_primary_position_z', 'true_primary_direction_x',
                                                                            'true_primary_direction_y','true_primary_direction_z']].values,dtype = torch.float)
        print('GRAPHS: %s / %s: %s'%(h,len(events), time.time() - start_g))
        graphs.append( Data(x = x, edge_index = edge_index,y=y.unsqueeze(0)))
        count = count + 1
    torch.save(graphs,'E:\\final_project\\data\\graphs_ran\\graphs_%s.pkl'%j)
    print((-start + time.time())/60)
    print('%s / %s' %(j+1, 30))
