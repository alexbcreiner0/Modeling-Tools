import numpy as np
from scipy.linalg import eig
import math

def random_weighted_graph(n, c = 1, weight_range = (1,10)):
    rng = np.random.default_rng()
    unweighted_graph = matrix_to_list(random_adj_matrix(n, c))
    graph = {u: {} for u in unweighted_graph}
    for u in graph:
        for v in unweighted_graph[u]:
            w = rng.integers(weight_range[0], weight_range[1])
            graph[u][v] = int(w)
    return graph

def matrix_to_list(adj_matrix):
    n = len(adj_matrix)
    vertices = list(range(0,n))
    G = {vertices[i]: [] for i in range(n)}
    for i,u in enumerate(G):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                G[u].append(vertices[j])
    return G

def random_adj_matrix(n, c=1):
    p = math.log(n) / n
    adj_matrix = [[0]*n for i in range(n)]
    for i in range(n):
        for j in range(n):
            flip = np.random.binomial(1, p)
            # print(flip)
            adj_matrix[i][j] = flip
    return adj_matrix

def is_weighted(G):
    if len(G) == 0: return False
    for adj in G.values():
        if type(adj) == dict: return True
        break
    return False

def unweightify(G):
    new_G = {v: {} for v in G}
    for v in new_G:
        new_G[v] = [u for u in G[v].keys()]
    return new_G

def reverse(G):
    rev_G = {u: [] for u in G}
    for u in G:
        for v in G[u]:
            rev_G[v].append(u)
    return rev_G

def connectedness(G, node_order = None):

    if is_weighted(G):
        G = unweightify(G)
    ccnum, comp_id = 0, {u: 0 for u in G}
    visited = {u: False for u in G}

    def explore(u, node_order= None):
        visited[u] = True
        comp_id[u] = ccnum
        for v in G[u]:
            if not visited[v]:
                explore(v)

    if node_order == None:
        node_order = list(G.keys())

    for u in node_order:
        if not visited[u]:
            ccnum += 1
            explore(u)

    return ccnum, comp_id

def pre_post(G):
    
    visited = {u: False for u in G}
    pre, post = {u: None for u in G}, {u: None for u in G}
    clock = 0

    def explore(u):
        nonlocal clock
        visited[u] = True
        clock += 1
        pre[u] = clock

        for v in G[u]:
            if not visited[v]:
                explore(v)
        clock += 1
        post[u] = clock

    for u in G:
        if not visited[u]:
            explore(u)

    return pre, post

def order_by_post_desc(G):
    _, post = pre_post(G)
    return sorted(post, key=post.get, reverse=True)

def strong_connectedness(G):
    if is_weighted(G):
        G = unweightify(G)
    rev_G = reverse(G)
    order = order_by_post_desc(rev_G)
    return connectedness(G, order)

def is_strongly_connected(G):
    ccnum, _ = strong_connectedness(G)
    if ccnum == 1:
        return True
    else: 
        return False

def random_strongly_connected_weighted_graph(n):
    G = random_weighted_graph(n)
    while not is_strongly_connected(G):
        G = random_weighted_graph(n)
    return G

def weighted_graph_to_matrix(G):
    n = len(G)
    matrix = [np.zeros(n) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if j in G[i]:
                matrix[i][j] = G[i][j]
    return np.array(matrix)

def productivize(matrix, epsilon=1e-2):
    evals, _ = eig(matrix)
    index = np.argmax(evals.real)
    if evals[index] >= 1:
        matrix /= (np.abs(evals[index]) + epsilon)
    return matrix

def random_irreducible_productive_matrix(inputs):
    print(inputs)
    dim = int(inputs["dim"])
    G = random_strongly_connected_weighted_graph(dim)
    return productivize(weighted_graph_to_matrix(G))

if __name__ == "__main__":
    G = random_strongly_connected_weighted_graph(4)
    print(G)
    matrix = weighted_graph_to_matrix(G)
    print(matrix)
    print(productivize(matrix))

