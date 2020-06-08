# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import time
from torch_geometric.data import Data



## BELOW 30 .pkl FILES ARE CREATED, EACH CONTAINING 10.000 GRAPHS IN THE CONGA-LINE EDGE CONFIGURATION.
## THE FILES ARE CREATED BY LOADING THE PADDED .csv EVENT FILES CREATED EARLIER.
scalar =  pd.read_csv('C:\\applied_ML\\final_project\\data\\scalar.csv')
scalar = scalar.loc[:,scalar.columns[1:len(scalar.columns)]]

for j in range(0,30):
    graphs = list()
    start = time.time()
    count = 1
    events  = pd.Series.reset_index(scalar['event_no'][0 + j*10000: 10000 + j*10000], drop = True)
    for h in range(0,len(events)):
        event = pd.read_csv('E:\\final_project\\data\\events_padded\\%s.csv'%events[h])
        upper = event.index.values.tolist()
        lower = np.roll(upper,1).tolist()
        edge_index = torch.tensor([upper,
                                           lower], dtype = torch.long)
        x = torch.tensor(event.values.tolist(),dtype = torch.float)
        y = torch.tensor(scalar.loc[h,['true_primary_energy','true_primary_time',
                                                                            'true_primary_position_x','true_primary_position_y',
                                                                            'true_primary_position_z', 'true_primary_direction_x',
                                                                            'true_primary_direction_y','true_primary_direction_z']].values,dtype = torch.float)
        graphs.append( Data(x = x, edge_index = edge_index,y=y.unsqueeze(0)))
        count = count + 1
    torch.save(graphs,'E:\\final_project\\data\\graphs\\graphs_%s.pkl'%j)
    print((-start + time.time())/60)
    print('%s / %s' %(j+1, 30))
