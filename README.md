
<h1><strong>Neutrino Reconstruction using Graph Neural Networks</strong></h1>
<p>This file is a quick, technical walkthrough of the final project 'Neutrino Reconstruction using Graph Neural Networks' by Peter Bagnegaard and Rasmus F. &Oslash;rs&oslash;e as part of the Big Data course at NBI 2020.</p>
<p><strong>Note:</strong> To run these python files you'll need the 120000.db, 140000.db and 160000.db source data files, which are NOT included in this github repo.</p>

<p><strong>The python files contains additional comments.</strong></p>

The slides from the presentation can be found here: https://drive.google.com/file/d/1X7bwemvhyGfqq_EpK_2X8CQLL47hiE2z/view?usp=sharing


<h2>Understanding the data (\tools\DisplayEvents.py)</h2>
<p>&nbsp;</p>
<p>&nbsp;</p>

The event data consists of 5 primary values: dom_x, dom_y, dom_z, which defines a position in the Ice Cube detector grid. In addition, there's the dom_charge and dom_time, which is the measured electrical charge of the dom and the time at which the measurement was made.

To get a more intuitive feel, we've made a plotting function that displays the doms and their location on their strings. You'll find this under \tools\DisplayEvents.py . The script reads the source data files and displays the doms that measured something for the specified event. The plots are 3D, and the colorbar labels time, so purple dots are early measurements and yellowish dots represents some of the last measurements made during the specified events. Points are scaled via the dom_charge variable. There's also the possibility of getting the plots as a sequence, such that you see the points as they are measured. To do this, set the optinal parameter real_time = True.

<h2>Building the Graphs</h2>

Graph Neural Networks adds an extra layer of complexity to your data. In order to use the torch_geometric package, which is the one you want to use for graph neural networking, then you must take your custom data and transform it to the torch_geometric data format. 

Since the torch_geometric library is quite young, there's not a whole lot of blogs out there that describes in great detail how to compile custom datasets into the torch_geometric framework.  We've read just about every article out there, like https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8, which emphasizes that
the right way to go about this is to define a custom dataset class for your dataset, such that it can be accessed by the machinery in torch_geometric. We failed to do this. 

As mentioned in the torch_geometric documentation (https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html), the dataset-class is not strictly needed in order for your data to comply with the torch_geometric formalism. You can do batching as long as your data is structured as a list of Data-objects. This is the route we took and therefore the one we'll explain below.

To build a single graph you need two things: 
1) Node Features, called x. x must be structured as [N_nodes,N_node_features]
2) Edges, called edge_index. The edge_index is a torch.tensor containing the information regarding how the nodes are connected.  

Take a look the code below:
```html
    x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
``` 
This defines 4 nodes with two node features each. Now one must define how these are connected:
```html
    edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]], dtype=torch.long)
``` 
<p>This edge_index says that: <br />node 0 is connected to node 1.<br />node 1 is connected to node 0.<br />node 2 is connected to node 1<br />node 0 is connected to node 3<br />node 3 is connected to node 2</p>

This can now be compiled into the Data-format via:
```html
    from torch_geometric.data import Data
    graph = Data(x = x, edge_index = edge_index)
``` 
This is now considered a graph by torch_geometric. In total we have:
```html
    from torch_geometric.data import Data
    x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 0, 3],
                   [1, 0, 1, 3, 2]], dtype=torch.long)
    graph = Data(x = x, edge_index = edge_index)
```
If one wanted to make batch of graphs for training, one could do so by first making a list()-object containing Data-objects like graph in the code above. To turn it into batches, we need to use torch_geometric.data.DataLoader :

```html
    from torch_geometric.data import Data
    from torch_geometric.data import DataLoader
    x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 0, 3],
                   [1, 0, 1, 3, 2]], dtype=torch.long)
    graph = Data(x = x, edge_index = edge_index)
    
    list_of_Data_objects = [graph]*64                                               # Repeats graph 64 times in list
    
    my_desired_batch_size = 32                                                      # you got this
    loader = DataLoader(list_of_Data_objects, batch_size = my_desired_batch_size)   # Reads your graphs into DataLoader-format
    
    my_batch_of_graphs = next(iter(loader))                                         # Accesses the next batch in the DataLoader
    
```

However, before we could use this we needed to identify all events and start building the x-tensors for each graph. Eventually we settled on creating the graphs from a 'bare' template where all doms were present in the graph but would by default have dom_time = dom_charge = 0. Then for each event we wanted to graph, we would then change the values of the corresponding nodes in the template and save this as a .csv file. This way, the majority of the nodes would have dom_time = dom_charge = 0, expect the nodes that actually measured something during that event. The reason for this was that we realized that the fact that some doms does not measure something during an event is information on it's own, which is something that the GNN should (hopefully) pick up on. Another more technical reason is that, through some early experimentation with the library, we found that torch_geometric doesn't handle graphs with a varying number of nodes very well, so we had to produce graphs that had a fixed number of nodes. You'll find the script defining our x-tensors this way under \tools\WriteEventsToCsv.py . This script relies on csv-files generated from \tools\MergeDatabses.py, which just merge the three different source-files into two csv-files, one for training data (sequential.csv) and one for target-values (scalar.csv). In hindsight, saving all of these events (150.000) as csv-files might not have been the smartest thing to do. Because each event now had many more nodes associated to it, the 150.000 x-tensor csv-files takes up approx. 60gb of space. This is due to the fact that most events originally had around 40 - 100 rows of data, but after adding these to the bare template, each x-tensor had around 5200 rows, where most, except position, is zero. 

....but, after inspecing the .csv files (which took forever to be generated) from \tools\WriteEventsToCsv.py, we painfully realized that not all of the x-tensors had the same amount of nodes. This was due to the fact that during some events, one dom might measure something more than once. To overcome this, we chose to 0-pad the x-tensors such that they were all of equal length. This is done in tools\PadEvents.py, which reloads all the .csv files from \tools\WriteEventsToCsv.py and adds padding to them, and saves them as seperate files. 

At this point we were ready to create the edge_index tensors. Initially we wanted to mimic a distance-based scheme proposed in https://arxiv.org/pdf/1902.07987.pdf, which deals with graph neural networks for particle reconstruction in colliders with irregular geometries. However, we quickly realized that using a distance based scheme were too computationally heavy, as we couldn't create the edge_index-tensors quicker than 4-ish seconds. At 150.000 graphs, this would have taken forever. This was a bummer, as we had hoped such a edge_index would enhance the GNN's ability to determine position and direction. 

We eventually settled on a simple but consistent edge_index scheme: The conga-line! Every node would be connected once in the order in which their indicies came in the x-tensors. This had the advantage of being very quick and that the edge_index would be build in a consistent way, that wouldn't change from graph to graph. The conga-line edge_index is calculated in \tools\CreateCongaGraphs.py that also reads the x-tensors and pickles the Data-objects as graphs, such that they are ready for training.

To test our initial thought that the edge_index needed to be consistent across graphs, we tried to develop the worst possible edge_index: A random one! Instead of assigning the nodes edges in a consistent way, we would, for each node in a single graph, generate 4 random connections to other nodes in the graph. This way, the resulting edge_index would be quite random in graphs and across graphs. This is done in \tools\CreateRandomGraphs.py.

<h2>The Graph Neural Network</h2>

This was our first time tinkering with the torch_geometric package. Models are built layer-wise, which adds a lot of freedom to tailor your model to your problem. It also means, however, that it is up to you as the programmer to make sure that the data is passed through layers that changes the dimensionality of the data such that it eventually becomes comparable with your target-values. Heres a rough breakdown:
1) Pooling layers changes the rows of the x-tensor (but by a fraction, which was the main reason why we thought the framework was difficult to use with graphs with a varying number of nodes). 
2) Convolutional layers changes the columns of the x-tensor (But allows you to set a specific size).

There exists other layers that changes these dimensions, one example of this is torch.nn.linear, which changes columns of the x-tensor.

After experimenting with a wide range of layers and configurations, we eventually settled for two models. 

1) The Linear Model
2) The Non-Linear Model

The linear model consists of three TopKPooling-layers and two torch.nn.linear-layers. Initially we had dropout-layers and f.relu-activation layers, but through testing we found that the dropout-layer decreased accuracy quite a bit and that the relu-activation function messed up predictions on position and direction. The non-linear model is configured quite similar, but has neural layers based on torch.nn.RNNCell. 

You can find the script running the linear model on the conga edge configuration at \models\LinearModel\LinearModelConga.py and the linear model on the random edge configuration at \models\LinearModel\LinearModelRan.py.

You can find the comparison of the edge configurations in the slides.












