#some list of graphs
import edmonds_utils
import sys
import math
from athityakumar_mst import Arc

ALPHA = 1
BETA = 0.5
graphs_to_practice_on = [
    {
    '1': {'3': 7, '2': 26, '4': 1},
    '0': {'1': 23, '3': 43, '2': 28, '4': 6},
    '3': {'1': 33, '2': 44, '4': 91},
    '2': {'1': 41, '3': 41, '4': 2},
    '4': {'1': 3, '3': 9, '2': 2},
    },
    {
    '1': {'8': 6, '3': 76, '2': 95, '5': 8, '6': 3},
    '0': {'1': 83, '8': 16, '6': 1},
    '3': {'8': 55, '5': 7, '4': 39, '7': 2, '6': 3},
    '2': {'3': 23},
    '5': {'8': 1, '2': 1, '4': 7, '7': 1, '6': 6},
    '4': {'8': 6, '5': 32},
    '7': {'1': 1, '2': 1, '5': 3, '4': 3, '6': 13, '8': 12},
    '6': {'1': 3, '2': 3, '5': 15, '4': 4, '7': 9, '8': 4},
    '8': {'1': 13, '3': 1, '5': 35, '4': 47, '7': 88, '6': 74},
    },
    {
    '20': {'21': 6},
    '22': {'13': 1, '21': 93, '1': 95, '3': 5, '2': 1, '7': 3},
    '1': {'10': 7, '12': 1, '20': 1, '17': 1, '18': 6, '15': 2, '3': 24, '2': 40, '5': 4, '7': 18, '6': 14, '8': 18},
    '0': {'22': 100},
    '3': {'15': 2, '18': 1, '1': 4, '2': 58, '5': 16, '4': 50, '7': 24, '6': 11, '8': 1},
    '2': {'10': 1, '3': 32, '4': 1, '7': 9, '6': 3, '8': 7},
    '5': {'8': 2, '3': 4, '4': 17, '7': 4, '6': 21},
    '4': {'8': 1, '5': 14, '7': 3, '18': 1},
    '7': {'10': 3, '13': 1, '3': 20, '5': 40, '4': 24, '6': 51, '8': 67},
    '6': {'10': 1, '18': 2, '3': 15, '5': 25, '4': 7, '7': 39, '8': 3},
    '9': {'10': 87, '13': 5, '12': 2, '20': 1, '14': 4, '17': 3, '18': 6, '15': 5},
    '8': {'9': 100, '10': 1, '15': 1, '18': 1},
    '11': {'13': 37, '12': 35, '15': 19, '14': 12, '17': 14, '16': 6, '18': 3},
    '10': {'11': 64, '13': 1, '12': 32, '15': 3, '14': 1, '18': 3},
    '13': {'1': 1, '12': 27, '15': 12, '14': 54, '17': 3},
    '12': {'11': 33, '13': 42, '15': 5, '21': 1, '17': 7, '18': 2, '8': 1, '14': 9},
    '15': {'11': 2, '13': 5, '12': 2, '14': 19, '17': 39, '16': 35, '18': 2, '2': 1, '5': 1, '4': 1},
    '14': {'11': 1, '13': 8, '12': 1, '15': 38, '17': 9, '16': 2, '18': 5},
    '17': {'18': 66, '15': 10, '14': 1, '16': 57},
    '16': {'18': 2, '15': 3, '17': 24},
    '19': {'20': 98},
    '18': {'19': 100},
    },
]


class Part:
    def __init__(self, list_of_arcs, wight):
        self.arcs = list_of_arcs
        self.w = wight

    def __repr__(self):
        return str([l for l in self.arcs])


class PartOld:
    # second_order_map = get
    def __init__(self, list_of_arcs):
        self.arcs = list_of_arcs
        self.w = self._w_of_part()

    def __repr__(self):
        return str([l for l in self.arcs])

    def _w_of_part(self):
        double_u = 0.0
        for (v, w, u) in self.arcs:
            double_u += w
        return double_u


def get_partsOLD(T, e):
    parts = []
    #first part is the arc itself
    e_tail, e_weight, e_head = e
    parts.append(Part([e], e_weight))
    for e_loop in T:
        tail, weight, head = e_loop
        if (tail == e_head and head != e_tail) or (head == e_tail and tail != e_head):
            parts.append(Part([e, e_loop]))
    return parts


def get_parts(T, e, second_order_map):
    parts = []
    #first part is the arc itself
    e_tail, e_weight, e_head = e
    parts.append(Part([e], e_weight))
    for e_loop in T:
        tail, weight, head = e_loop
        if (tail == e_head and head != e_tail):
            tup = (head, tail, e_head, e_tail)
            if tup in second_order_map:
                parts.append(Part([e, e_loop], wight=second_order_map[tup]))
            else:
                parts.append(Part([e, e_loop], wight=0))
        elif (head == e_tail and tail != e_head):
            tup = (e_head, e_tail, head, tail)
            if tup in second_order_map:
                parts.append(Part([e, e_loop], wight=second_order_map[tup]))
            else:
                parts.append(Part([e, e_loop], wight=0))
    return parts


# if (a_id == b_head and a_head != b_id):
#     tup = (a_head, a_id, b_head, b_id)
#     if tup not in my_map:
#         my_map[tup] = 0
#     my_map[tup] += 1
# if (a_head == b_id and a_id != b_head):
#     tup = (b_head, b_id, a_head, a_id)
#     if tup not in my_map:
#         my_map[tup] = 0



def get_lost_parts(lost_arcs, G, second_order_map=None):
    lost_parts = []
    for arc in lost_arcs:
        lost_parts += get_parts(G, arc, second_order_map=second_order_map)
    return lost_parts


def get_incoming_arcs(e, rest_arcs):
    """
    :param e: an edge
    :param rest_arcs: list of arcs
    :return: incoming_arcs, set with edge directed towards the tail of given 'e' (the first parameter)
    """
    incoming_arcs = set()
    e_tail, e_weight, e_head = e
    for arc in rest_arcs:
        tail, weight, head = arc
        if tail == e_tail and head != e_head:
            incoming_arcs.add(arc)
    return incoming_arcs


def probability_edge(e, T, alpha):
    """
    :param e: an edge
    :param T: a Graph
    :return: probability, numerator / denominator
    """
    e_set = set()
    e_set.add(e)
    T_union = T.union(e_set)
    e_tail, e_weight, e_head = e
    weights = []
    for tail, weight, head in T_union:
        if tail == e_tail:
            weights.append(weight)
    numerator = math.exp(alpha * e_weight)
    denominator = sum([math.exp(alpha * ww) for ww in weights])
    if denominator == 0:
        return 0
    return float(numerator)/denominator


def probability_part(part, T, alpha):
    p_parts_list = []
    for edge in part.arcs:
        p_parts_list.append(probability_edge(edge, T, alpha))
    p = reduce(lambda x, y: x * y, p_parts_list)
    return p


def probability_part_given_T(part, T, alpha, print_me=None):
    numerator = probability_part(part, T, alpha)
    denominator_components = []
    T_CUT_WITH_PART = T & set(part.arcs)

    for edge in part.arcs:
        denominator_components.append(
            probability_edge(edge, T_CUT_WITH_PART, alpha)
        )

    denominator = reduce(lambda x, y: x * y, denominator_components)
    if denominator == 0:
        if print_me:
            print 'probability_part_given_T'
        return 0
    if print_me:
        print 'non zero'
    return float(numerator) / denominator


def cyclic(graph):
    """Return True if the directed graph has a cycle.
    The graph must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False

    """
    visited = set()
    path = [object()]
    path_set = set(path)
    stack = [iter(graph)]
    while stack:
        for v in stack[-1]:
            if v in path_set:
                return True
            elif v not in visited:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                stack.append(iter(graph.get(v, ())))
                break
        else:
            path_set.remove(path.pop())
            stack.pop()
    return False


def is_cycle(list_of_arcs):
    my_dict = {}
    for arc in list_of_arcs:
        tail, weight, head = arc
        if head not in my_dict:
            my_dict[head] = []
        my_dict[head].append(tail)
    return cyclic(my_dict)


def get_cycle_arcs(e, T, G):
    """
    general - line 7 in the second algorithm.
    go for edge in g:
        if union(e,T,edge) for a cycle
        put edge in cycle arcs list.
    :param e: an edge
    :param T: a set of arcs
    :param G: a set of arcs
    :return: set of arcs
    """
    cycle_arcs = set()
    for arc in G:
        arc_with_possible_cycle = set()
        arc_with_possible_cycle.add(e)
        arc_with_possible_cycle.add(arc)
        arc_with_possible_cycle = arc_with_possible_cycle.union(T)
        if is_cycle(arc_with_possible_cycle):
            cycle_arcs.add(arc)
    return cycle_arcs


def tails_counter_method(grph):
    tails_counter = {}
    for arc in grph:
        tail, weight, head = arc
        if tail not in tails_counter:
            tails_counter[tail] = 0
        tails_counter[tail] += 1
    return tails_counter


def mst_2nd_order(G, N, alpha=None, beta=None, second_order_map=None):
    """
    :param G: Graph
    :param N: Number of words in sentence / number of vertices
    :return: T - mst second order
    """
    if not alpha:
        alpha = ALPHA
    if not beta:
        beta = BETA

    T = set()
    # for i in range(1, N):
    for i in range(N):
        # print "ITER: {0}".format(i)
        # print "Number of vertices remained: {0}".format(len(G))
        gain_dict = {}
        loss_dict = {}
        cycle_dict = {}
        lost_arcs_dict = {}

        ### if there is a vertex, which only one edge connect to him, we'll add him
        tails_counter = tails_counter_method(G)
        signle_arc = None
        if 1 in tails_counter.values():
            single_tails_indexes = [key for key in tails_counter if tails_counter[key] == 1]

            single_tail = single_tails_indexes[-1]
            for arc in G:
                tail, weight, head = arc
                if tail == single_tail:
                    signle_arc = arc
                    break

        if signle_arc:
            rest_arcs = set(G) - set(signle_arc)
            incomins_arcs = get_incoming_arcs(signle_arc, rest_arcs)
            cycle_arcs = get_cycle_arcs(signle_arc, T, G)
            lost_arcs = incomins_arcs.union(cycle_arcs)
            lost_arcs_dict[signle_arc] = lost_arcs
            arg_min = signle_arc

        else:
            for e in G:
                structure = """
                v is tail, w is weight, u is head, shape like that:
                  ______
                 |      |
                 V      |"""
                (v, w, u) = e
                PARTSe = get_parts(T, e, second_order_map=second_order_map)
                gain = 0.0
                for part in PARTSe:
                    my_set = set()
                    my_set.add(Arc(v, w, u))
                    united_T = my_set.union(T)
                    gain += part.w * probability_part_given_T(part, united_T, alpha, print_me=False)
                gain_dict[e] = gain

                # we calculated gain, now we'll calculate loss:
                rest_arcs = set(G) - set(e)
                incomins_arcs = get_incoming_arcs(e, rest_arcs)
                cycle_arcs = get_cycle_arcs(e, T, G)
                cycle_dict[e] = cycle_arcs
                lost_arcs = incomins_arcs.union(cycle_arcs)
                lost_arcs_dict[e] = lost_arcs
                lost_parts = get_lost_parts(lost_arcs, G, second_order_map=second_order_map)
                loss = 0.0
                for part in lost_parts:
                    loss += part.w * probability_part_given_T(part, T, alpha, print_me=False)
                loss_dict[e] = loss

            # time to decide which arc is joining T:
            value_dict = {}
            for e in loss_dict:
                value_dict[e] = beta * loss_dict[e] - (1-beta) * gain_dict[e]
            arg_min = min(value_dict, key=value_dict.get)

        add_to_T = set()
        add_to_T.add(arg_min)
        T = T.union(add_to_T)

        # remove lost arcs:
        for arc in lost_arcs_dict[arg_min]:
            G.remove(arc)

        G.remove(arg_min)

    return T


def check_different_alphas():
    my_graph = graphs_to_practice_on[1]
    my_graph = edmonds_utils.dict_graph_2_arc_graph(my_graph)
    print my_graph
    first = mst_2nd_order(my_graph, 4, 4, 0.5)
    a_graph = graphs_to_practice_on[1]
    print a_graph
    second = mst_2nd_order(edmonds_utils.dict_graph_2_arc_graph(a_graph), 4, 7, 0)
    print first == second


def check_get_parts():


    my_graph = {
    '1': {'2': 7, '3': 8},
    '2': {'1': 41, '3': 41},
    '3': {'1': 33},
    }
    my_graph = edmonds_utils.dict_graph_2_arc_graph(my_graph)
    a = Arc(tail='2', weight=5, head='1')
    for part in get_parts(my_graph, a):
        print part
    print mst_2nd_order(my_graph, 2)

def check():
    # my_graph = {
    # '1': {'2': 7, '3': 8},
    # '2': {'1': 41, '3': 41},
    # '3': {'1': 33},
    # }
    my_graph = {
    '1': {'3': 7, '2': 4, '4': 1},
    '0': {'1': 23, '3': 43, '2': 28, '4': 6},
    '3': {'1': 33, '2': 44, '4': 91},
    '2': {'1': 41, '3': 41, '4': 2},
    '4': {'1': 3, '3': 9, '2': 2},
    }
    my_graph = edmonds_utils.graph_to_inverse_values(my_graph)
    my_graph = edmonds_utils.dict_graph_2_arc_graph(my_graph)

    a = Arc(tail='2', weight=6, head='1')
    b = Arc(tail='4', weight=6, head='2')
    print probability_part_given_T(Part(list_of_arcs=[a, b]), set(my_graph), 1)
    print probability_part(Part(list_of_arcs=[a, b]), set(my_graph), 1)
    # print probability_edge(a, set(my_graph), 0)

if __name__ == "__main__":
    check()
