# -*- coding: utf-8 -*-
import pandas as pd
import time
scalar =  pd.read_csv('C:\\applied_ML\\final_project\\data\\scalar.csv')
scalar = scalar.loc[:,scalar.columns[1:len(scalar.columns)]]

events = scalar['event_no'][0:150000]
count = 1
now = time.time()
for event in events:
    x = pd.read_csv('E:\\final_project\\data\\events\\%s.csv'%event)
    x = x[x.columns[1:len(x.columns)]]
    pad = pd.DataFrame(x.loc[0,:]).T
    pad.loc[0,:] = 0
    if len(x) > 5624:
        x.to_csv(r'E:\\final_project\\data\\events_prob\\%s.csv'%event, index = False)
    if len(x) == 5624:
        x.to_csv(r'E:\\final_project\\data\\events_padded\\%s.csv'%event, index = False)
    if len(x)< 5624:
        len_diff = abs(len(x) - 5624)
        for k in range(0,len_diff):
            x = x.append(pad)
        x.to_csv(r'E:\\final_project\\data\\events_padded\\%s.csv'%event, index = False)
    print('%s/%s' %(count,len(events)))
    count = count + 1

print('Time Passed %s' %(time.time() - now)/3600)