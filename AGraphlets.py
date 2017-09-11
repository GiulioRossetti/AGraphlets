import networkx as nx
import itertools
import tqdm
import backboning
import os
import numpy as np
import json
import sys
from networkx.readwrite import json_graph


__author__ = 'rossetti'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"


class AGraphlet(object):

    def __init__(self, network_filename, node_attr=None, backbone_threshold=0.3, approx_percentile=70):
        self.network_filename = network_filename
        self.backbone_threshold = backbone_threshold
        self.approx_percentile = approx_percentile
        self.g = nx.Graph()
        self.node_attr = node_attr is not None

        if not os.path.exists("res"):
            os.makedirs("res")

        if self.node_attr:
            self.__add_node_attributes(node_attr)

    def __add_node_attributes(self, node_attr):
        f = open(node_attr)
        for l in f:
            l = l.rstrip().split("\t")
            self.g.add_node(int(l[0]), {"name": l[1]})

    def __backbone(self):
        # Read graph and compute backbone
        table, nnodes, nnedges = backboning.read(self.network_filename, "w", undirected=True)
        nc_table = backboning.noise_corrected(table)
        nc_backbone = backboning.thresholding(nc_table, self.backbone_threshold)
        backboning.write(nc_backbone, "res/simplified", "nc", ".")
        f = open("res/simplified_nc.csv")
        # Load filtered graph
        for l in f:
            l = l.split("\t")
            try:
                self.g.add_edge(int(l[0]), int(l[1]))
            except ValueError:
                pass
        os.remove("res/simplified_nc.csv")

    @staticmethod
    def __compute_subsets(s, m):
        return set(itertools.combinations(s, m))

    def __local_components_size_percentile(self):
        sizes = []

        for n in self.g:
            eg = nx.ego_graph(self.g, n, center=False)

            # for each connected component in the ego-minus-ego network
            for comp in nx.connected_components(eg):
                size = len(comp)
                sizes.append(size)

        pv = np.percentile(np.array(sizes), self.approx_percentile)
        return pv

    def __approx_graphlets_identification(self, max_comp_size=20, min_pattern_size=3, max_pattern_size=5):
        size_to_graph = {}
        c = 0

        # cycle over graph nodes
        for n in tqdm.tqdm(self.g, desc="Nodes"):
            eg = nx.ego_graph(self.g, n, center=False)
            c += 1

            # for each connected component in the ego-minus-ego network
            for comp in nx.connected_components(eg):

                # greedy approximation
                if len(comp) > max_comp_size:
                    continue

                # cycle over graphlet size
                for x in range(min_pattern_size-1, max_pattern_size):

                    # candidate ego nodes subsets for "graphlet" of size x
                    subsets = self.__compute_subsets(comp, x)

                    if len(subsets) > 0:

                        for mk in tqdm.tqdm(subsets):
                            ext_mk = list(mk)
                            ext_mk.append(n)
                            nodes = tuple(sorted(ext_mk))

                            # Reintroducing ego
                            mkg = nx.subgraph(eg, mk)
                            nds = [n]
                            nds.extend(list(mkg.nodes()))
                            mkg.add_star(nds)

                            x_class = None
                            if self.node_attr:
                                mkg.node[n]["name"] = self.g.node[n]["name"]
                                x_class = mkg.node[n]["name"]

                            # Reindex node ids
                            mkg = nx.convert_node_labels_to_integers(mkg, first_label=0)
                            n_edges = mkg.number_of_edges()

                            if x not in size_to_graph:
                                size_to_graph[x] = {n_edges: [[mkg, {nodes: None, "xcl": x_class}]]}
                            else:
                                # Check if already present
                                isomorphic = False
                                if n_edges in size_to_graph[x]:
                                    for i in range(0, len(size_to_graph[x][n_edges])):

                                        # subset already seen: avoid isomorphism
                                        if nodes in size_to_graph[x][n_edges][i][1]:
                                            continue

                                        if x_class is not None:
                                            # set central node class
                                            if x_class != size_to_graph[x][n_edges][i][1]["xcl"]:
                                                continue

                                        graphlet, count = size_to_graph[x][n_edges][i]
                                        if nx.is_isomorphic(graphlet, mkg, node_match=self.__compare):
                                            isomorphic = True
                                            size_to_graph[x][n_edges][i][1][nodes] = None
                                            break

                                    if not isomorphic:
                                        size_to_graph[x][n_edges].append([mkg, {nodes: None, "xcl": x_class}])
                                else:
                                    size_to_graph[x][n_edges] = [[mkg, {nodes: None, "xcl": x_class}]]

        return size_to_graph

    def __compare(self, a, b):
        if self.node_attr:
            return a['name'] == b['name']
        else:
            return True

    @staticmethod
    def __save_patterns(rs):
        match = open("res/subgraph_match.csv", "w")
        stats = open("res/stats.csv", "w")
        stats.write("pid,nodes,edges,instances\n")
        pc = 0
        for size in rs:
            for nedges in rs[size]:
                for pattern, instances in rs[size][nedges]:
                    data = json_graph.node_link_data(pattern)
                    jd = json.dumps(data)
                    f = open("res/pattern-id%s-n%s-e%s-i%s.json" %
                                      (pc, pattern.number_of_nodes(), pattern.number_of_edges(), len(instances)), "w")
                    f.write(jd)
                    f.flush()
                    f.close()
                    stats.write("%s,%s,%s,%s\n" % (pc, pattern.number_of_nodes(), pattern.number_of_edges(),
                                                   len(instances)))
                    for i in instances:
                        match.write("%s\t%s\n" % (pc, str(i)))
                    pc += 1
        match.flush()
        match.close()

        stats.flush()
        stats.close()

    def execute(self, min_pattern_size=3, max_pattern_size=5):

        # Compute network backbone
        self.__backbone()

        # Save filtered graph in json
        data = json_graph.node_link_data(self.g)
        jd = json.dumps(data)
        f = open("res/simplified_graph.json", "w")
        f.write(jd)
        f.flush()
        f.close()

        # Identify percentile component size (tune approximation)
        psize = int(self.__local_components_size_percentile())
        print("Selected max component size: %s" % psize)

        # Compute subgraphs
        rs = self.__approx_graphlets_identification(max_comp_size=psize,
                                                    min_pattern_size=min_pattern_size,
                                                    max_pattern_size=max_pattern_size)

        # Export Graphlets
        self.__save_patterns(rs)


if __name__ == "__main__":
    import argparse

    sys.stdout.write("-------------------------------------\n")
    sys.stdout.write("            {AGraphlet}              \n")
    sys.stdout.write("   Approximate Graphlet Extraction   \n")
    sys.stdout.write("-------------------------------------\n")
    sys.stdout.write("Author: " + __author__ + "\n")
    sys.stdout.write("Email:  " + __email__ + "\n")
    sys.stdout.write("------------------------------------\n")

    parser = argparse.ArgumentParser()

    parser.add_argument('network_file', type=str, help='network file (weighted edge list format)')
    parser.add_argument('percentile', type=int, help='component size percentile', default=70)
    parser.add_argument('backbone_threshold', type=float, help='backbone filtering threshold', default=0.3)
    parser.add_argument('min_graphlet_size', type=int, help='minimum graphlet size', default=3)
    parser.add_argument('max_graphlet_size', type=int, help='max graphlet size', default=5)
    parser.add_argument('-a', '--node_attr', type=str, help='node attribute file', default=None)

    args = parser.parse_args()

    ag = AGraphlet(args.network_file, node_attr=args.node_attr,
                   approx_percentile=args.percentile, backbone_threshold=args.backbone_threshold)
    ag.execute(min_pattern_size=args.min_graphlet_size, max_pattern_size=args.max_graphlet_size)

