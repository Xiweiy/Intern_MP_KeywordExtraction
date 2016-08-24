import matplotlib
matplotlib.use('Agg')
%matplotlib inline   ##for ipython notebook

import matplotlib.pyplot as plt
plt.figure(3,figsize=(18,18)) 

graph = networkx.Graph()
for i in bow[:30]:
    sentence_to_graph(graph, i)

pos = networkx.spring_layout(graph)
networkx.draw_networkx_edges(graph, pos, alpha=0.3, width = 3, length = 100, edge_color='m')
networkx.draw_networkx_nodes(graph, pos, alpha = 0.4, node_size = 100, node_color = 'w')
networkx.draw_networkx_edges(graph, pos, alpha =0.4, node_size =100 , length=100, width = 1 , edge_color ='k')
networkx.draw_networkx_labels(graph, pos, fontsize=100)
plt.show()
