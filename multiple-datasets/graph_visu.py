import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class visualization(object):
    def __init__(self, corr_estimate_pairwise, corr_estimate,  x_corr, num_signals, num_dataset, type=1):
        """

        Args:
            corr_estimate (): matrix of correlation between pairs of datasets
            x_corr ():
            num_signals ():
        """
        self.graph_matrix = corr_estimate_pairwise
        self.typ2graph_matrix = corr_estimate
        self.edge_list = x_corr
        self.signals = num_signals
        self.num_nodes = num_dataset
        self.type = type

    def create_subplots(self, shape):
        fig, axes = plt.subplots(*shape)
        self.axes = axes.flatten()

    def create_graph_nodes(self):
        node_list = list(range(self.num_nodes))
        g_list = [None] * self.signals
        for i in range(self.signals):
            g_list[i] = nx.Graph()
            g_list[i].add_nodes_from(node_list)
            for j in range(len(self.edge_list)):
                if self.graph_matrix[i, j] == 1:
                    g_list[i].add_edge(*self.edge_list[j])
            # pos = nx.random_layout(g_list[i], seed=89)
            pos = nx.circular_layout(g_list[i], scale=0.8)
            #print(pos)
            self.axes[i].set_title('signal ' + str(i + 1))
            self.axes[i].set_xlim([-1.25, 1.25])
            self.axes[i].set_ylim([-1.25, 1.25])
            nx.draw_networkx(g_list[i], pos=pos, with_labels=True, font_weight='bold', ax=self.axes[i])
        if self.signals % 2 > 0:
            self.axes[-1].set_visible(False)
        plt.show()

    def generate_coordinates(self, node_list, datasets=1):
        y = np.linspace(1, -1, self.signals + 1)
        x = np.linspace(-0.9, 0.9, self.num_nodes)
        offset_x = np.linspace(-0.9, 0,9, self.num_nodes)

        #xy = map(list, zip(x, y))
        #pos = dict(zip(range(self.signals + 1), xy))
        pos = []
        for xi in x:
            px = [xi]*(self.signals+1)
            xy = map(list, zip(px, y))
            pos.append(dict(zip(range(self.signals + 1), xy)))

        return pos




        return pos

    def create_type2_graph(self):
        node_list = list(range(self.signals))
        g_list = [None] * self.num_nodes
        #glist = nx.Graph()
        #glist.add_nodes_from(node_list)

        pos_all = self.generate_coordinates(node_list, 1)

        for i in range(self.num_nodes):
            g_list[i] = nx.Graph()
            g_list[i].add_nodes_from(node_list)





    def visualize(self):
        if self.type == 1:
            self.create_subplots((int(np.ceil(self.signals / 2)), 2))
            self.create_graph_nodes()

        if self.type == 2:
            self.create_subplots((self.signals, 1))
            self.create_type2_graph()


# g = nx.Graph()
# g.add_node(2)
# g.add_node(3)
# g.add_node(4)
# g.add_node(5)
#
# g.add_edge(2, 3)
# g.add_edge(2, 4)
# g.add_edge(4, 5)
# print(nx.info(g))
# h = nx.Graph()
# h.add_node(1)
# h.add_node(2)
# h.add_node(3)
# h.add_edge(2, 3)
#
# fig, axes = plt.subplots(2)
# ax = axes.flatten()
# nx.draw_networkx(g, with_labels=True, font_weight='bold', ax=ax[0])
#
# nx.draw_networkx(h, with_labels=True, font_weight='bold', ax=ax[1])
# plt.show()
