
Neutrino Reconstruction using Graph Neural Networks


This file is a quick, technical walkthrough of the final project 'Neutrino Reconstruction using Graph Neural Networks' by Peter Bagnegaard and Rasmus F. Ørsøe as part of the Big Data course at NBI 2020. 

Note: To run these python files you'll need the 120000.db, 140000.db and 160000.db source data files, which are NOT included in this github repo.

Understanding the data (\tools\DisplayEvents.py)

The event data consists of 5 primary values: dom_x, dom_y, dom_z, which defines a position in the Ice Cube detector grid. In addition, there's the dom_charge and dom_time, which is the measured electrical charge of the dom and the time at which the measurement was made.

To get a more intuitive feel, we've made a plotting function that displays the doms and their location on their strings. You'll find this under \tools\DisplayEvents.py . The script reads the source data files and displays the doms that measured something for the specified event. The plots are 3D, and the colorbar labels time, so purple dots are early measurements and yellowish dots represents some of the last measurements made during the specified events. Points are scaled via the dom_charge variable. There's also the possibility of getting the plots as a sequence, such that you see the points as they are measured. To do this, set the optinal parameter \textbf{real_time} to True


