import networkx as nx
import matplotlib.pyplot as plt


class visualization(object):
    def __init__(self, corr_estimate, x_corr, num_signals, num_dataset):
        """

        Args:
            corr_estimate (): matrix of correlation between pairs of datasets
            x_corr ():
            num_signals ():
        """
        self.graph_matrix = corr_estimate
        self.edge_list = x_corr
        self.signals = num_signals
        self.num_nodes = num_dataset

    def create_subplots(self):
        fig, axes = plt.subplots(4, 2)
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
            self.axes[i].set_title('signal '+str(i))
            nx.draw_networkx(g_list[i], with_labels=True,font_weight='bold', ax=self.axes[i])

        plt.show()


    def visualize(self):
        self.create_subplots()
        self.create_graph_nodes()





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
