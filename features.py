from itertools                          import product
from utils                              import get_paths

from networkx.generators.ego            import ego_graph
from networkx.exception                 import NetworkXError
from networkx.algorithms.simple_paths   import all_simple_paths
from networkx.utils                     import pairwise
from scipy.spatial                      import distance
from math                               import sqrt, log

import numpy as np
import settings


# Store the generated context subgraphs to avoid computing them multiple times
# (this is done separately on each thread, and so it consumes more memory as
# more threads are used, however, it is cleaned up automatically when the worker
# threads die)
context_subgraphs = {}

def get_feature_vector(graph, triple, relations, remove_triple=False, original_positive=None, centrality_indices=None, rels_to_study=None):
    res = []
    s, r, t = triple
    recyprocal_removed = False
    rng = range(1, settings.MAX_CONTEXT_SIZE + 1)

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
        res += get_intersection_feats(s, t, ents_s, ents_t, graph, True, centrality_indices=centrality_indices)

    ###########################################################################

    # Reachable entities for all relations
    for i, j in product(rng, rng):
        triples_s = [(s_g, r_g, t_g) for s_g, t_g, r_g in context_subgraphs[(s, i)].edges.data("rel")]
        triples_t = [(s_g, r_g, t_g) for s_g, t_g, r_g in context_subgraphs[(t, j)].edges.data("rel")]

        for rel in relations:
            ents_s = [t_g for s_g, r_g, t_g in triples_s if r_g == rel] + [s]
            ents_t = [t_g for s_g, r_g, t_g in triples_t if r_g == rel] + [t]
            res += get_intersection_feats(s, t, ents_s, ents_t, graph, centrality_indices=centrality_indices)

    ###########################################################################

    # Path-based features
    if settings.USE_PATHS:
        rels_dict = {rel: i for i, rel in enumerate(rels_to_study)}
        for i in rng:
            triples = [(s, r, t) for s, t, r in context_subgraphs[(s, i)].edges.data("rel")
                        if r in rels_to_study]
            paths = get_paths(triples, s, t, i)
            matrix = np.zeros((len(rels_to_study),) * i)

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

def get_intersection_feats(source, target, ls1, ls2, graph, add_non_cond_only=False, centrality_indices=None):
    s1 = set(ls1)
    s2 = set(ls2)
    union = s1 | s2
    inter = s1 & s2

    len_s1 = float(len(s1))
    len_s2 = float(len(s2))
    len_union = float(len(union))
    len_inter = float(len(inter))

    jacc = len_inter / len_union
    overlap = len_inter / min(len_s1, len_s2)
    sorensen_dice = 2 * len_inter / (len_s1 + len_s2)
    cosine = len_inter / (sqrt(len_s1) * sqrt(len_s2)) 

    res = [len_s1, len_s2, len_inter, jacc, overlap, sorensen_dice, cosine]

    if add_non_cond_only:
        adamic_adar = 0.0
        for node in inter:
            try:
                adamic_adar += 1 / log(graph.degree(node))
            except ZeroDivisionError:
                # Degree = 1
                adamic_adar += 1.0
            except ValueError:
                # Degree = 0
                continue

        cent1 = float(centrality_indices[source]) if centrality_indices else 0.0
        cent2 = float(centrality_indices[target]) if centrality_indices else 0.0
        res += [adamic_adar, cent1, cent2]

    return res


def get_header(relations, rels_to_study):
    if not rels_to_study:
        rels_to_study = relations

    header = []
    header.append("triple")
    header.append("label")
    rng = range(1, settings.MAX_CONTEXT_SIZE + 1)

    # Regular subgraph features
    for i, j in product(rng, rng):
        header.append(f"numEnts_src_c{i}_1ctx")
        header.append(f"numEnts_tgt_c{j}_1ctx")
        header.append(f"numEnts_inter_c{i}-c{j}_2ctx")
        header.append(f"jaccard_c{i}-c{j}_2ctx")
        header.append(f"overlap_c{i}-c{j}_2ctx")
        header.append(f"sorensen-dice_c{i}-c{j}_2ctx")
        header.append(f"cosine_c{i}-c{j}_2ctx")
        header.append(f"adamic-adar_c{i}-c{j}_2ctx")
        header.append(f"centrality_src_1ctx")
        header.append(f"centrality_tgt_1ctx")

    # Reachability-based features
    for i, j in product(rng, rng):
        for rel in relations:
            r = rel.replace(" ", "-")
            header.append(f"numEnts_rel-{r}_src_c{i}_1ctx")
            header.append(f"numEnts_rel-{r}_tgt_c{j}_1ctx")
            header.append(f"numEnts_rel-{r}_inter_c{i}-c{j}_2ctx")
            header.append(f"jaccard_rel-{r}_c{i}-c{j}_2ctx")
            header.append(f"overlap_rel-{r}_c{i}-c{j}_2ctx")
            header.append(f"sorensen-dice_rel-{r}_c{i}-c{j}_2ctx")
            header.append(f"cosine-{r}_c{i}-c{j}_2ctx")

    # Path-based features
    if settings.USE_PATHS:
        rels_clean = [x.replace(" ", "-") for x in rels_to_study]
        rels_dict = dict(enumerate(rels_clean))
        rng_rels = range(len(rels_to_study))

        for i in rng:
            matrix = np.zeros((len(rels_to_study),) * i, dtype=object)
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