from settings                           import MAX_CONTEXT_SIZE
from itertools                          import product
from utils                              import get_paths

from networkx.generators.ego            import ego_graph
from networkx.exception                 import NetworkXError
from networkx.algorithms.simple_paths   import all_simple_paths
from networkx.utils                     import pairwise

import numpy as np

# Store the generated context subgraphs to avoid computing them multiple times
# (this is done separately on each thread, and so it consumes more memory as
# more threads are used, however, it is cleaned up automatically when the worker
# threads die)
context_subgraphs = {}

def get_feature_vector(graph, triple, relations, remove_triple=False, original_positive=None):
    res = []
    s, r, t = triple
    recyprocal_removed = False
    rng = range(1, MAX_CONTEXT_SIZE + 1)

    # Remove the triple itself from the graph if it's a positive example
    # and the original positive if it's a negative one
    # (and any reciprocal relations)
    if remove_triple:
        graph.remove_edge(s, t, key=r)
        try:
            graph.remove_edge(t, s, key=r)
            recyprocal_removed = True
        except NetworkXError: pass
    elif original_positive:
        o_s, o_r, o_t = original_positive
        graph.remove_edge(o_s, o_t, key=o_r)
        try:
            graph.remove_edge(o_t, o_s, key=o_r)
            recyprocal_removed = True
        except NetworkXError: pass

    ###########################################################################

    # Load the subgraphs if they are not there yet
    for i in rng:
        if (s, i) not in context_subgraphs:
            context_subgraphs[(s, i)] = ego_graph(graph, s, i)
        if (t, i) not in context_subgraphs:
            context_subgraphs[(t, i)] = ego_graph(graph, t, i)

    ###########################################################################

    # Regular subgraph features
    for i, j in product(rng, rng):
        ents_s = list(context_subgraphs[(s, i)].nodes) + [s]
        ents_t = list(context_subgraphs[(t, j)].nodes) + [t]
        res += get_intersection_feats(ents_s, ents_t)

    ###########################################################################

    # Reachable entities for all relations
    for i, j in product(rng, rng):
        triples_s = [(s_g, r_g, t_g) for s_g, t_g, r_g in context_subgraphs[(s, i)].edges.data("rel")]
        triples_t = [(s_g, r_g, t_g) for s_g, t_g, r_g in context_subgraphs[(t, j)].edges.data("rel")]

        for rel in relations:
            ents_s = [t_g for s_g, r_g, t_g in triples_s if r_g == rel] + [s]
            ents_t = [t_g for s_g, r_g, t_g in triples_t if r_g == rel] + [t]
            res += get_intersection_feats(ents_s, ents_t)

    ###########################################################################

    # Path-based features
    rels_dict = {rel: i for i, rel in enumerate(relations)}
    for i in rng:
        triples = [(s, r, t) for s, t, r in context_subgraphs[(s, i)].edges.data("rel")]
        paths = get_paths(triples, s, t, i)
        matrix = np.zeros((len(relations),) * i)

        ### TODO ad-hoc code --- probably refactor this in the future
        for path in paths:
            if i == 1:
                matrix[rels_dict[path[0][1]]] += 1
            elif i == 2:
                matrix[rels_dict[path[0][1]]][rels_dict[path[1][1]]] += 1
            elif i == 3:
                matrix[rels_dict[path[0][1]]][rels_dict[path[1][1]]][rels_dict[path[2][1]]] += 1

        total_paths = np.sum(matrix)
        res.append(total_paths)
        res += matrix.flatten().tolist()


    ###########################################################################
    
    # Restore the deleted edges
    if remove_triple:
        graph.add_edge(s, t, rel=r, key=r)
        if recyprocal_removed:
            graph.add_edge(t, s, rel=r, key=r)
    elif original_positive:
        o_s, o_r, o_t = original_positive
        graph.add_edge(o_s, o_t, rel=o_r, key=o_r)
        if recyprocal_removed:
            graph.add_edge(o_t, o_s, rel=o_r, key=o_r)

    ###########################################################################
    
    # Done
    return res

def get_intersection_feats(ls1, ls2):
    s1 = set(ls1)
    s2 = set(ls2)
    union = s1 | s2
    inter = s1 & s2
    jacc = float(len(inter)) / len(union)
    return [float(len(s1)), float(len(s2)), float(len(inter)), jacc]

def get_header(relations):
    header = []
    header.append("triple")
    header.append("label")
    rng = range(1, MAX_CONTEXT_SIZE + 1)

    # Regular subgraph features
    for i, j in product(rng, rng):
        header.append(f"numEnts_src_c{i}")
        header.append(f"numEnts_tgt_c{j}")
        header.append(f"numEnts_inter_c{i}-c{j}")
        header.append(f"jaccard_c{i}-c{j}")

    # Reachability-based features
    for i, j in product(rng, rng):
        for rel in relations:
            r = rel.replace(" ", "-")
            header.append(f"numEnts_rel-{r}_src_c{i}")
            header.append(f"numEnts_rel-{r}_tgt_c{j}")
            header.append(f"numEnts_rel-{r}_inter_c{i}-c{j}")
            header.append(f"jaccard_rel-{r}_c{i}-c{j}")

    # Path-based features
    rels_clean = [x.replace(" ", "-") for x in relations]
    rels_dict = dict(enumerate(rels_clean))
    rng_rels = range(len(relations))

    for i in rng:
        matrix = np.zeros((len(relations),) * i, dtype=object)
        if i == 1:
            for x in rng_rels:
                matrix[x] = f"path_{rels_dict[x]}_c1"
        elif i == 2:
            for x, y in product(rng_rels, rng_rels):
                matrix[x][y] = f"path_{rels_dict[x]}->{rels_dict[y]}_c2"
        elif i == 3:
            for x, y, z in product(rng_rels, rng_rels, rng_rels):
                matrix[x][y][z] = f"path_{rels_dict[x]}->{rels_dict[y]}->{rels_dict[z]}_c3"
        
        header.append(f"totalPaths_c{i}")
        header += matrix.flatten().tolist()

    return header