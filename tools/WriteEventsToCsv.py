import pandas as pd
from math import sqrt
import torch
from copy import deepcopy
from time import time 

## LOADS RAW EVENT DATA
sequential = pd.read_csv('C:\\applied_ML\\final_project\\data\\sequential.csv')
sequential = sequential.loc[:,sequential.columns[1:len(sequential.columns)]]
## LOADS TARGET VARIABLES
scalar =  pd.read_csv('C:\\applied_ML\\final_project\\data\\scalar.csv')
scalar = scalar.loc[:,scalar.columns[1:len(scalar.columns)]]
## LOADS THE FULL BUT EMPTY GRAPH.
bare_graph  =  pd.read_csv('C:\\applied_ML\\final_project\\data\\bare_graph.csv')

check_list = pd.Series(pd.DataFrame(sequential[['dom_x','dom_y','dom_z']])[['dom_x','dom_y','dom_z']].apply(tuple, axis=1).unique())


### BELOW, THE FIRST 150.000 EVENTS ARE IDENTIFIED AND PUT INTO THE 'BARE GRAPH' FORMAT AND SAVED AS A .CSV-file.
##  NOTE!!:: THIS IS ABOUT 60gb WORTH OF DATA
events = scalar['event_no'][0:150000]
start_time = time()
count = 0
for event in events:
    index = sequential['event_no'] == event
    seq_data = pd.DataFrame(sequential.loc[index,:]).reset_index(drop = True)
    seq_check_list = pd.DataFrame(seq_data[['dom_x','dom_y','dom_z']])[['dom_x','dom_y','dom_z']].apply(tuple, axis=1).reset_index(drop=True)
    x = deepcopy(bare_graph)
    for pulse in range(0,len(seq_data)):
        pulse_location = check_list[check_list == seq_check_list[pulse]].index[0]
        
        if(type(x.loc[pulse_location, 'dom_time']) != pd.core.series.Series):
            if (x.loc[pulse_location, 'dom_time']) == 0:
                x.loc[pulse_location, ['dom_charge','dom_time','SRTInIcePulses']] = seq_data.loc[pulse,['dom_charge','dom_time','SRTInIcePulses']]
            else:
                x = x.append(seq_data.loc[pulse,['dom_x','dom_y','dom_z','dom_charge','dom_time','SRTInIcePulses']])
        else:    
            x = x.append(seq_data.loc[pulse,['dom_x','dom_y','dom_z','dom_charge','dom_time','SRTInIcePulses']])
    print('%s / %s' %(count + 1, len(events)))
    count = count  +1  
    x.to_csv(r'E:\\final_project\\data\\events\\%s.csv'%event)
print('Total runtime (hours): %s' %((time() -  start_time)/3600))