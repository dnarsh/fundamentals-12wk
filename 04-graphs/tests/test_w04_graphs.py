import random

import pytest

from _support import load

S = load(__file__)

DIAMOND = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
CYCLIC = {1: [2], 2: [3], 3: [1, 4], 4: []}


def all_nodes(g):
    nodes = set(g)
    for vs in g.values():
        nodes.update(vs)
    return nodes


def random_graph(rng, n, m, weighted=False, dag=False, wmax=10):
    g = {i: [] for i in range(n)}
    for _ in range(m):
        u, v = rng.randrange(n), rng.randrange(n)
        if dag:
            if u == v:
                continue
            u, v = min(u, v), max(u, v)
        g[u].append((v, rng.randint(0, wmax)) if weighted else v)
    return g


def bellman_ford(g, src):
    dist = {src: 0}
    for _ in range(len(all_nodes(g))):
        changed = False
        for u, es in g.items():
            if u in dist:
                for v, w in es:
                    if v not in dist or dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        changed = True
        if not changed:
            break
    return dist


# ---------- BFS / DFS ----------

def test_bfs_order_diamond():
    assert S.bfs_order(DIAMOND, "a") == ["a", "b", "c", "d"]


def test_dfs_order_diamond_first_listed_neighbour_first():
    assert S.dfs_order(DIAMOND, "a") == ["a", "b", "d", "c"]


def test_orders_terminate_on_cycles_and_visit_once():
    assert S.bfs_order(CYCLIC, 1) == [1, 2, 3, 4]
    assert S.dfs_order(CYCLIC, 1) == [1, 2, 3, 4]


def test_orders_only_include_reachable_nodes():
    g = {"a": ["b"], "b": [], "z": ["a"]}
    assert S.bfs_order(g, "a") == ["a", "b"]
    assert S.dfs_order(g, "a") == ["a", "b"]


def test_start_is_neighbour_only_node():
    g = {"a": ["b"]}
    assert S.bfs_order(g, "b") == ["b"]
    assert S.dfs_order(g, "b") == ["b"]


def test_unknown_start_raises_keyerror():
    with pytest.raises(KeyError):
        S.bfs_order(DIAMOND, "nope")
    with pytest.raises(KeyError):
        S.dfs_order(DIAMOND, "nope")


def test_self_loop_and_duplicate_edges():
    g = {1: [1, 2, 2], 2: [1]}
    assert S.bfs_order(g, 1) == [1, 2]
    assert S.dfs_order(g, 1) == [1, 2]


def test_dfs_preorder_goes_deep_before_wide():
    g = {1: [2, 3], 2: [4], 4: [5], 3: []}
    assert S.dfs_order(g, 1) == [1, 2, 4, 5, 3]


def test_dfs_survives_deep_path():
    n = 5000
    g = {i: [i + 1] for i in range(n)}
    g[n] = []
    assert S.dfs_order(g, 0) == list(range(n + 1))
    assert S.bfs_order(g, 0) == list(range(n + 1))


def test_shortest_hops():
    assert S.shortest_hops(DIAMOND, "a") == {"a": 0, "b": 1, "c": 1, "d": 2}
    g = {1: [2, 3], 2: [4], 3: [4], 4: [5], 6: [1]}
    assert S.shortest_hops(g, 1) == {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}


@pytest.mark.parametrize("seed", range(5))
def test_hops_consistent_with_bfs_order_on_random_graphs(seed):
    rng = random.Random(seed)
    g = random_graph(rng, 40, 90)
    hops = S.shortest_hops(g, 0)
    order = S.bfs_order(g, 0)
    assert set(order) == set(hops) and len(order) == len(set(order))
    # BFS discovers nodes in nondecreasing hop distance
    assert [hops[v] for v in order] == sorted(hops[v] for v in order)
    for u in hops:
        for v in g.get(u, []):
            assert hops[v] <= hops[u] + 1


# ---------- topological sort ----------

def test_toposort_diamond_valid():
    order = S.topological_sort(DIAMOND)
    assert sorted(order) == ["a", "b", "c", "d"]
    assert order.index("a") < order.index("b") < order.index("d")
    assert order.index("a") < order.index("c") < order.index("d")


def test_toposort_includes_neighbour_only_nodes_and_empty_graph():
    assert sorted(S.topological_sort({"a": ["b"]})) == ["a", "b"]
    assert S.topological_sort({}) == []


@pytest.mark.parametrize("g", [CYCLIC, {1: [1]}, {1: [2], 2: [1]}])
def test_toposort_cycle_raises(g):
    with pytest.raises(ValueError):
        S.topological_sort(g)


@pytest.mark.parametrize("seed", range(8))
def test_toposort_random_dags(seed):
    rng = random.Random(seed)
    g = random_graph(rng, 30, 70, dag=True)
    perm = list(range(30))
    rng.shuffle(perm)
    g = {perm[u]: [perm[v] for v in vs] for u, vs in g.items()}  # hide the natural order
    order = S.topological_sort(g)
    assert sorted(order) == sorted(all_nodes(g))
    pos = {v: i for i, v in enumerate(order)}
    for u, vs in g.items():
        for v in vs:
            assert pos[u] < pos[v]


def test_toposort_deep_chain():
    n = 5000
    g = {i: [i + 1] for i in range(n)}
    assert S.topological_sort(g) == list(range(n + 1))


# ---------- Dijkstra ----------

WG = {
    "s": [("a", 7), ("b", 2)],
    "b": [("a", 3), ("c", 8)],
    "a": [("c", 1)],
    "c": [],
}


def test_dijkstra_known_graph():
    assert S.dijkstra(WG, "s") == {"s": 0, "b": 2, "a": 5, "c": 6}


def test_dijkstra_unreachable_absent_and_source_zero():
    g = {1: [(2, 4)], 2: [], 3: [(1, 1)]}
    d = S.dijkstra(g, 1)
    assert d == {1: 0, 2: 4}


def test_dijkstra_zero_weights_and_parallel_edges():
    g = {1: [(2, 5), (2, 0), (3, 0)], 2: [(3, 0)], 3: [(1, 0)]}
    assert S.dijkstra(g, 1) == {1: 0, 2: 0, 3: 0}


def test_dijkstra_float_weights():
    g = {1: [(2, 0.5), (3, 2.0)], 2: [(3, 0.25)], 3: []}
    d = S.dijkstra(g, 1)
    assert d[3] == pytest.approx(0.75)


def test_dijkstra_negative_weight_raises():
    with pytest.raises(ValueError):
        S.dijkstra({1: [(2, -1)], 2: []}, 1)


def test_dijkstra_unknown_source_raises_keyerror():
    with pytest.raises(KeyError):
        S.dijkstra(WG, "zzz")


@pytest.mark.parametrize("seed", range(10))
def test_dijkstra_matches_bellman_ford(seed):
    rng = random.Random(seed)
    g = random_graph(rng, 25, 70, weighted=True)
    assert S.dijkstra(g, 0) == bellman_ford(g, 0)


def test_shortest_path_known():
    assert S.shortest_path(WG, "s", "c") == (6, ["s", "b", "a", "c"])
    assert S.shortest_path(WG, "s", "s") == (0, ["s"])
    assert S.shortest_path(WG, "c", "s") is None


@pytest.mark.parametrize("seed", range(10))
def test_shortest_path_is_a_real_path_with_reported_cost(seed):
    rng = random.Random(100 + seed)
    g = random_graph(rng, 20, 60, weighted=True)
    dist = bellman_ford(g, 0)
    for target in range(20):
        res = S.shortest_path(g, 0, target)
        if target not in dist:
            assert res is None
            continue
        cost, path = res
        assert cost == dist[target]
        assert path[0] == 0 and path[-1] == target
        total = 0
        for u, v in zip(path, path[1:]):
            total += min(w for x, w in g[u] if x == v)  # cheapest parallel edge
        assert total == cost
