
Neutrino Reconstruction using Graph Neural Networks


This file is a quick, technical walkthrough of the final project 'Neutrino Reconstruction using Graph Neural Networks' by Peter Bagnegaard and Rasmus F. Ørsøe as part of the Big Data course at NBI 2020. 

Note: To run these python files you'll need the 120000.db, 140000.db and 160000.db source data files, which are NOT included in this github repo.

Understanding the data (\tools\DisplayEvents.py)

The event data consists of 5 primary values: dom_x, dom_y, dom_z, which defines a position in the Ice Cube detector grid. In addition, there's the dom_charge and dom_time, which is the measured electrical charge of the dom and the time at which the measurement was made.

To get a more intuitive feel, we've made a plotting function that displays the doms and their location on their strings. You'll find this under \tools\DisplayEvents.py . The script reads the source data files and displays the doms that measured something for the specified event. The plots are 3D, and the colorbar labels time, so purple dots are early measurements and yellowish dots represents some of the last measurements made during the specified events. Points are scaled via the dom_charge variable. There's also the possibility of getting the plots as a sequence, such that you see the points as they are measured. To do this, set the optinal parameter real_time = True.

Building the Graphs

Graph Neural Networks adds an extra layer of complexity to your data. In order to use the torch_geometric package, which is the one you want to use for graph neural networking, then you must take your custom data and transform it to the torch_geometric data format. 

Since the torch_geometric library is quite young, there's not a whole lot of blogs out there that describes in great detail how to compile custom datasets into the torch_geometric framework.  We've read just about every article out there, like https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8, which emphasizes that
the right way to go about this is to define a custom dataset class for your dataset, such that it can be accessed by the machinery in torch_geometric. We failed to do this. 

As mentioned in the torch_geometric documentation (https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html), the dataset-class is not strictly needed in order for your data to comply with the torch_geometric formalism. You can do batching as long as your data is structured as a list of Data-objects. This is the route we took and therefore the one we'll explain below.

To build a single graph you need three things: 
1) Node Features, called x. x must be structured as [N_nodes,N_node_features]
2) Edges, called edge_index. The edge_index is a torch.tensor containing the information regarding how the graphs are connected.  

Take a look the code below:
```html
    x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
``` 
This defines 4 nodes with two node features each. Now one must define how these are connected:
```html
    edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]], dtype=torch.long)
``` 
This edge_index says that: 
a) node 0 is connected to node 1
b) node 1 is connected to node 0
c) node 2 is connected to node 1
d) node 0 is connected to node 3
e) node 3 is connected to node 2


<h2>Example of code</h2>

<pre>
    <div class="container">
        <div class="block two first">
            <h2>Your title</h2>
            <div class="wrap">
            //Your content
            </div>
        </div>
    </div>
</pre>
```




