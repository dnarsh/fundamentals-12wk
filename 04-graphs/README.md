# Week 04: BFS / DFS / Dijkstra on a dependency graph

## Goal
Model "package A must be built before package B" as a directed graph and answer questions about it: visit order, hop distance, build order, cheapest path.

## Resource
MIT 6.006 Spring 2020, graph search and shortest-path lectures: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/

## Representation
- Unweighted: `dict[node, list[node]]`, edge `u -> v` means "u must come before v". A node that only appears as a neighbour is still a node (it has no outgoing edges).
- Weighted: `dict[node, list[tuple[node, weight]]]`.
- Nodes are hashable and mutually orderable only in tests that say so; do not sort nodes unless a rule below says so.

## Implement in `solution.py`
- `bfs_order(graph, start) -> list`: visit order, neighbours in listed order. `start` not in graph and not a neighbour anywhere: `KeyError`.
- `dfs_order(graph, start) -> list`: preorder, neighbours tried in listed order (the first listed neighbour is explored first). Iterative or recursive, but it must survive a path of 5000 nodes (hint: Python's recursion limit).
- `shortest_hops(graph, start) -> dict`: node -> minimum number of edges; unreachable nodes are absent.
- `topological_sort(graph) -> list`: every node (including neighbour-only nodes) exactly once, each edge `u -> v` has `u` before `v`. A cycle raises `ValueError`.
- `dijkstra(graph, source) -> dict`: node -> cheapest cost; unreachable nodes absent. Any negative weight raises `ValueError`. You may use `heapq` or your week 3 heap.
- `shortest_path(graph, source, target) -> tuple | None`: `(cost, [source, ..., target])`, or `None` if unreachable; `source == target` gives `(0, [source])`.

## Behaviours the tests check
- Exact orders on small hand-drawn graphs, including a diamond and a cycle.
- BFS/DFS visit each reachable node once and never loop on cycles.
- Topological order property on random DAGs, `ValueError` on cycles and self-loops.
- Dijkstra agrees with a Bellman-Ford oracle (in the test) on random graphs; zero-weight edges; parallel edges; the returned path really costs what is reported.
- Deep path graph does not overflow the stack.

## Aloud-questions (no notes)
1. Why does BFS give shortest paths in an unweighted graph and DFS does not?
2. Why does Dijkstra fail with negative edges? Give a 3-node counterexample.
3. What is the running time of your Dijkstra in terms of V and E, and what data structure sets it?
4. How does topological sort detect a cycle in your implementation?
5. When would you use DFS postorder instead of Kahn's algorithm, and what would you lose?

## What I got wrong
