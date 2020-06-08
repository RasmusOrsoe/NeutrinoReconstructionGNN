import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm


def DisplayEvent(event_no,real_time = False,pause = 0.5):
    ## INPUT: A single event_number
    ## OUTPUT: Plots xyz scatter plot with colorebar indicating time.
    ## KARGS: real_time - default = False. If real_time == True, the plot will be a sequentially updated
    ##        plot. 
    ##        pause - the time between each sequential update. Default = 0.5s.  
    index = sequential_small['event_no'] == event_no
    if real_time == False:
        first_event = pd.DataFrame(sequential_small.loc[index,:])
        
        marker_size = np.interp(first_event['dom_charge'], (first_event['dom_charge'].min()
                                                        , first_event['dom_charge'].max()), (10, 100))
        fig = plt.figure()
        ax = Axes3D(fig)
        img = ax.scatter(first_event['dom_x'].values,first_event['dom_y'].values,first_event['dom_z'].values, 
               c= (first_event['dom_time'].values),s = marker_size)
        
        plt.colorbar(img,cmap = plt.get_cmap('gnuplot'))
        ax.set_xlabel('X-dom')
        ax.set_ylabel('Y-dom')
        ax.set_zlabel('Z-dom')
        ax.grid(False)
        for k in range(0,len(first_event['dom_x'])):
            ax.plot([first_event['dom_x'].values[k],first_event['dom_x'].values[k]]
                ,[first_event['dom_y'].values[k],first_event['dom_y'].values[k]],[-800,800], c='grey')
    if real_time == True:
        first_event = pd.DataFrame(sequential_small.loc[index,:])
        
        marker_size = np.interp(first_event['dom_charge'], (first_event['dom_charge'].min()
                                                        , first_event['dom_charge'].max()), (10, 100))
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.grid(False)
        ax.set_xlabel('X-dom')
        ax.set_ylabel('Y-dom')
        ax.set_zlabel('Z-dom')
        ax.set_ylim([-800,800])
        ax.set_xlim([-800,800])
        ax.set_zlim([-800,800])
        color = first_event['dom_time'].values
        color = np.interp(color, (color.min(), color.max()), (0, +1))
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in color]
        for k in range(0,len(first_event['dom_x'])):
            ax.plot([first_event['dom_x'].values[k],first_event['dom_x'].values[k]]
                ,[first_event['dom_y'].values[k],first_event['dom_y'].values[k]],[-800,800], c='grey')
        for k in range(0,len(first_event['dom_x'])):
            plt.pause(pause)
            plt.ion()
            img = ax.scatter(first_event['dom_x'].values[k],first_event['dom_y'].values[k],first_event['dom_z'].values[k], 
                  s = marker_size[k], c = colors[k],cmap = cmap)
            fig.canvas.draw()
        m = cm.ScalarMappable(cmap=cmap)
        m.set_array(first_event['dom_time'].values)
        plt.colorbar(m)
    return;

## IMPORT DATA
db_file =  'C:\\applied_ML\\final_project\\data\\160000_00.db'
with sqlite3.connect(db_file) as con:
    query = 'select * from sequential'
    sequential = pd.read_sql(query, con)
    query = 'select * from scalar'
    scalar = pd.read_sql(query, con)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print('AVAILABLE TABLES:')
    print(cursor.fetchall())    
    
    
## SELECT SMALL TEST-BATCH
scalar_small = scalar.loc[0:1000,:]
sequential_small = pd.DataFrame()

for k in range(0,len(scalar_small)):
    index =  sequential['event_no'] == scalar_small['event_no'][k]
    sequential_small = sequential_small.append(sequential.loc[index,:])

## PICK EVENT AND DISPLAY
event= np.array(sequential_small['event_no'])[0]
DisplayEvent(event, real_time = False, pause = 0.1)
    


