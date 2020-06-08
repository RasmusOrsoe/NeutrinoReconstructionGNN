import torch                                      
import pandas as pd
import time
from torch.nn import MSELoss
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
import matplotlib.pyplot as plt
##############################################################################

start = time.time()                                                             ## STARTS TIMER FOR LATER USE

graphs = list()
for k in range(0,15):
    graphs.extend(['graphs_%s.pkl' %k])                                         ## LIST OF ALL GRAPH FILE NAMES
                                                                                #   ( FOR LOOPING LATER ON, IF NEEDED)
class Net(torch.nn.Module):                                                     #
    def __init__(self):                                                         #    
        super(Net, self).__init__()                                             #                                          
        self.pool1  = TopKPooling(5,ratio = 0.01)                               #
        self.nn1   = torch.nn.Linear(5,64)                                      # THESE ARE THE LAYERS OF THE MODEL                          
        self.pool2  = TopKPooling(64,ratio = 0.1)                               #
        self.pool3  = TopKPooling(64,ratio = 0.1)                               #            
        self.nn2   = torch.nn.Linear(64,8)                                      #
                                                                                # 
    def forward(self, data):                                                    #
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch      # THIS DEFINES HOW DATA MOVES BETWEEN LAYERS
        x, edge_index,_,batch,_,_ = self.pool1(x,edge_index,None,batch)         #                     
        x = self.nn1(x)                                                         # 
        x, edge_index,_,batch,_,_ = self.pool2(x,edge_index,None,batch)         #
        x, edge_index,_,batch,_,_ = self.pool3(x,edge_index,None,batch)         #    
        x = self.nn2(x)                                                         #
        return x                                           #    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # CHOOSING GPU IF AVAILABLE
model = Net().to(device)                                                        # MOUNTS MODEL ON DEVICE


batch_size = 32                                                                 #                                                                        
lr = 1e-3                                                                       # PARAMETERS FOR TRAINING AND PREDICTION
n_epochs = 20                                                                   #       ( lr = Learning Rate )

data_list = torch.load('E:\\final_project\\data\\graphs_ran\\%s' %graphs[0])    # GRABS FIRST Graph-FILE FOR TRAINING
                                                                                #   (You could loop over this )


                                                                                ## TRAINING
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)      # OPTIMIZER
loss_func = MSELoss()                                                           # LOSS-FUNCTION

loss_list =list()
loader = DataLoader(data_list, batch_size = batch_size)                         # LOADS THE Graph-file INTO THE BATCH FORMAT 
for i in range(0,len(loader)):                                                  # LOOP OVER BATCHES
    data_train = next(iter(loader))                                             # 
    data_train = data_train.to(device)                                          # MOUNTS DATA TO DEVICE
    model.train()                                                               #
    for epoch in range(n_epochs):                                               # LOOP OVER EPOCHS 
        optimizer.zero_grad()                                                   #
        out = model(data_train)                                                 # ACTUAL TRAINING
        loss = loss_func(out, data_train.y.float())                             #
        loss.backward()                                                         #
        optimizer.step()
        print('BATCH: %s / %s || EPOCH / %s : %s || MSE: %s' %(i, 
                                                               len(loader),
                                                               epoch, 
                                                               n_epochs
                                                               , loss.data.item()))                                                        #
        loss_list.append(loss.item())
print('TOTAL TRAINING TIME ON %s GRAPHS: %s' %(len(data_list),
                                               (time.time() - start)/60))

## PLOT LOSS
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.show()



                                                                                ## PREDICTION
data_list = torch.load('E:\\final_project\\data\\graphs_ran\\%s' %graphs[1])    # GRABS SECOND Graph-file FOR PREDICTION
loader = DataLoader(data_list, batch_size = batch_size)                         # LOADS THE Graph-file INTO THE BATCH FORMAT 
start = time.time()                                                             # TIMER FOR LATER
acc = 0                                                                         # VARIABLE TO HOLD NMAE CALCULATION
for i in range(0,len(loader)):                                                  # LOOP OVER GRAPHS IN FILE
    data_pred = next(iter(loader))
    model.eval()                                                                # PREDICTION AND CALCULATION OF NMAE-SCORE   
    data = data_pred.to(device)                                                 # 
    pred = model(data)                                                          #
    correct = data.y                                                            #
    acc = acc + (abs(pred - data.y)/abs(data.y)).detach().cpu().numpy()         #
    print('PREDICTING: %s /  %s' %(i,len(loader)))                                                                        
res = acc.sum(0)/(batch_size*len(loader))
print(res)



