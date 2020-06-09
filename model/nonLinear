import torch                                      
import pandas as pd
import numpy as np
import time
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn.models.graph_unet import GraphUNet
from torch.nn import Dropout
import matplotlib.pyplot as plt
##############################################################################

def NMAE(pred, data):
    correct = data.y                                                            #
    return (abs(pred - correct)/abs(correct)).detach().cpu().numpy()            #                                                                                #   ( KAN BRUGES TIL AT LOOPE OVER SENERE)

class Net(torch.nn.Module):                                                     
    def __init__(self):                                                         #    
#        self.nn1   = torch.nn.RNN(5,64)                                         # LAG I MODELLEN#                          
        super(Net, self).__init__()
        l1, l2, l3, l4 = 5, 64, 32, 8
                                             #                                            #
        self.relu  = torch.nn.ReLU  (inplace=True)                                # LAG I MODELLEN#                          
        self.pool1 = TopKPooling    (l1   ,ratio = 0.01)
        self.nn1   = torch.nn.RNNCell(l1,l2)                                     # LAG I MODELLEN#                          
        self.pool2 = TopKPooling    (l2   ,ratio = 0.1 )                                #
        self.nn2   = torch.nn.Linear(l2,l3)
        self.pool3 = TopKPooling    (l3   ,ratio = 0.1 )                                #
        self.nn3   = torch.nn.Linear(l3,l4)                                      #
                                                                                # 
    def forward(self, data):                                                    #
        # Get data
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch      # [179968, 5]             

        # 1: Pooling layer + Neural layer (NO activation)
        x, edge_index,_,batch,_,_ = self.pool1(x,edge_index,None,batch)         # [1824, 5]
        x = self.nn1(x)                                              # [1824, 64]

        # 2: Pooling layer + Neural layer (ReLU actication)
        x, edge_index,_,batch,_,_ = self.pool2(x,edge_index,None,batch)         # [192, 64]
        x = self.relu(self.nn2(x))                                              # [1824, 64]
        
        # 3: Pooling layer + Neural layer (NO activation)
        x, edge_index,_,batch,_,_ = self.pool3(x,edge_index,None,batch)         # 
        x = self.nn3(x)                                              # [192, 3]
        return x                                          #    

#%%
"""Initial values for training"""
n_epochs = 20
batch_size = 32
lr = 1e-3


start = time.time()                                                             ## STARTER EN COUNTER FOR AT MÅLE TID

graphs = list()
for k in range(0,15):
    graphs.extend(['graphs_%s.pkl' %k])                                         ## DETTE ER EN LISTE OVER ALLE FIL-NAVNENE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # VÆLGER GPU HVIS MULIGT
model = Net().to(device)
                                                                                # MOUNTER MODEL I GPU/CPU
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)    # OPTIMIZER-FUNCTION TIL TRÆNING
loss_func = MSELoss()                                                           # LOSS-FUNCTION TIL TRÆNING

data_list = torch.load('C:\\applied_ML\\final_project\\data\\%s' %graphs[0])        # HENTER 1 GRAF-FIL ( VI KAN LOOPE OVER DETTE SENERE)

#%%
"""
Train model
"""
# Initialize scaling and dataloader
scaler = StandardScaler()
loader = DataLoader(data_list, batch_size = batch_size)                         # LOADER DATA
# Lists for storing data during simulation

#NMAE_list = list()
export = list()
for j in range(0,10):
    loss_list = list()
    if j!= 0:
        del model
    model = Net().to(device)                                                                                # MOUNTER MODEL I GPU/CPU
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loader = DataLoader(data_list, batch_size = batch_size) 
    print(j)
    for i in range(0,len(loader)):                                                         # LOOP OVER BATCHES
        data_train = next(iter(loader))                                             # HIVER DATA UD AF DataLoader-FORMATET
        data_train = data_train.to(device)                                          #
        ## 'PRE'-PROCESSING
        if i == 0:
            scaler.fit(data_train.x.cpu().numpy()[:,0:5])                           
            data_train.x = torch.tensor(scaler.transform(data_train.x.cpu().numpy()[:,0:5]), dtype = torch.float).float().cuda()
        else:
            data_train.x = torch.tensor(scaler.transform(data_train.x.cpu().numpy()[:,0:5]), dtype = torch.float).float().cuda()
        #data_train.y = torch.tensor(data_train.y.cpu().numpy()[:,5:8],dtype=torch.float).float().cuda()
        model.train()                                                               #
        for epoch in range(n_epochs):                                               # LOOP OVER EPOCHS    # SELVE TRÆNINGEN   
            optimizer.zero_grad()                                                   #
            out = model(data_train)                                                 #
            loss = loss_func(out, data_train.y.float())                                     #
            loss.backward()                                                         #
            optimizer.step()
            print('BATCH: %s / %s || EPOCH / %s : %s || MSE: %s' %(i, len(loader), epoch, n_epochs, loss.data.item()))                                                        #
            loss_list.append(loss.item())
            #NMAE_list.append(NMAE(out, data_train))
        
    plt.clf()
    plt.plot(loss_list, '.')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title("Simulation nbr " + str(j))
    plt.pause(0.3)
    pd.DataFrame(loss_list).ato_csv(r'C:\\applied_ML\\final_project\\Long_sim\\sim%s.csv'%j, index = False)
    print('TOTAL TRAINING TIME ON %s GRAPHS: %s' %(len(data_list), (time.time() - start)/60))

    
    """
    Predict on data_pred
    """
    data_list = torch.load('C:\\applied_ML\\final_project\\data\\%s' %graphs[1])
    loader = DataLoader(data_list, batch_size = batch_size)
    start = time.time()                                                             #
    acc = 0
    for i in range(0,len(loader)):                                                                         #
        data_pred = next(iter(loader))
        data_pred.x = torch.tensor(scaler.transform(data_pred.x.cpu().numpy()[:,0:5]),dtype=torch.float).float().cuda()
        #data_pred.y = torch.tensor(data_pred.y.cpu().numpy()[:,5:8],dtype=torch.float).float().cuda()
        
        model.eval()                                                                # PREDICTION OG UDREGNING AF NMAE-SCORE   
        data = data_pred.to(device)                                                      #    ( BØR SKRIVES OM SENERE )
        pred = model(data)                                                          #
        correct = data.y                                                            #
        pred = model(data)                                                          #
        acc = acc + NMAE(pred, data)         #
    #    acc = acc + (abs(pred - data.y)/abs(data.y)).detach().cpu().numpy()         #
        print('PREDICTING: %s /  %s' %(i,len(loader)))                                                                        
    res = acc.sum(0)/(batch_size*len(loader))
    print(res)
    export.append(res)
pd.DataFrame(export).to_csv(r'C:\\applied_ML\\final_project\\nmae.csv', index = False)    
