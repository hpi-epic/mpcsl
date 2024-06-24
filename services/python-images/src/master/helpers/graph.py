from itertools import combinations, product


# I wish this was more legible, but it is ported over from pcalg::has.new.coll which is a complete mess...
def creates_collider(graph, x, fixed_parents, bi_parents, combination):
    res = False
    if len(combination) > 0 and True:
        if len(fixed_parents) > 0 and True:
            res = any([not (graph.has_edge(a, b) or graph.has_edge(b, a))
                       for a, b in product(fixed_parents, combination)])
        if not res and len(combination) > 1:
            res = any([not (graph.has_edge(a, b) or graph.has_edge(b, a)) for a, b in combinations(combination, 2)])
    if not res and len(combination) < len(bi_parents):
        excluded_bi_parents = list(bi_parents - set(combination))
        crazy_shit = set()
        for par in excluded_bi_parents:
            crazy_shit |= {n for n in graph.predecessors(par)
                           if not graph.has_edge(par, n)}  # No bidirectional edges
        papa = crazy_shit - {x}
        if len(papa) > 0:
            res = any([not (graph.has_edge(x, e) or graph.has_edge(e, x)) for e in papa])
    return res


def get_potential_confounders(graph, cause_node_id):
    all_parents = set(graph.predecessors(cause_node_id))
    bi_parents = set([node for node in all_parents if graph.has_edge(cause_node_id, node)])
    fixed_parents = all_parents - bi_parents

    possible_parent_combinations = []
    for i in range(len(bi_parents) + 1):
        for comb in combinations(bi_parents, i):
            if not creates_collider(graph, cause_node_id, fixed_parents, bi_parents, comb):
                possible_parent_combinations.append(list(fixed_parents) + list(comb))
    return possible_parent_combinations
