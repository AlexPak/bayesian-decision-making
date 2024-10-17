import matplotlib.pyplot as plt
import networkx as nx

def plot_bayesian_network(bn):
    labels = {i: var.name for i, var in enumerate(bn.vars, start=1)}
    pos = nx.spring_layout(bn.graph)
    nx.draw(bn.graph, pos, labels=labels, with_labels=True, node_color='lightblue', font_weight='bold')
    edge_labels = nx.get_edge_attributes(bn.graph, 'weight')
    nx.draw_networkx_edge_labels(bn.graph, pos, edge_labels=edge_labels)
    plt.show()