import sys
from collections import defaultdict, deque

INF = 10**9

gateways: list[str]
gateway_hallways: list[tuple[str, str]]
removed_hallways: set[tuple[str, str]]
graph: defaultdict[str, list[str]]

def _build_graph(edges: list[tuple[str, str]]) -> defaultdict[str, list[str]]:
    graph: defaultdict[str, list[str]] = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

def _reached_gateway(virus_pos: str) -> bool:
    return virus_pos.isupper()

def _get_target_gateway(virus_pos: str) -> str | None:
    best_gate = None
    best_dist = INF
    dist_from_virus = _get_dist(virus_pos)
    for gateway in gateways:
        dist = dist_from_virus[gateway]
        if dist == INF:
            continue
        if dist < best_dist or \
            (dist == best_dist and (best_gate is not None and gateway < best_gate)):
            best_gate = gateway
            best_dist = dist
    return best_gate


def _get_dist(start: str) -> dict[str, int]:
    dist: dict[str, int] = {v: INF for v in graph}
    dist[start] = 0
    queue = deque([start])
    while len(queue) > 0:
        current = queue.popleft()
        for neighbour in graph[current]:
            if (neighbour, current) in removed_hallways or \
                (current, neighbour) in removed_hallways:
                continue
            if dist[neighbour] == INF:
                dist[neighbour] = dist[current] + 1
                queue.append(neighbour)
    return dist

def _next_node(virus_pos: str, target_gateway: str) -> str:
    dist_from_gateway = _get_dist(target_gateway)
    candidates: list[str] = list()
    for neighbour in graph[virus_pos]:
        if (neighbour, virus_pos) in removed_hallways or \
            (virus_pos, neighbour) in removed_hallways:
            continue
        if dist_from_gateway[virus_pos] == dist_from_gateway[neighbour] + 1:
            candidates.append(neighbour)
    return min(candidates)

def _get_disconnected_hallways() -> list[tuple[str, str]]:
    global removed_hallways, gateway_hallways

    virus_pos = 'a'
    removed_hallways = set()
    
    def dfs(virus_pos: str) -> list[tuple[str, str]] | None:
        if _reached_gateway(virus_pos):
            return None
        target = _get_target_gateway(virus_pos)
        if target is None:
            return []
        for gate, node in [
            hallway for hallway in gateway_hallways if hallway not in removed_hallways
        ]:
            removed_hallways.add((gate, node))
            new_target = _get_target_gateway(virus_pos)
            if new_target is None:
                return [(gate, node)]
            new_pos = _next_node(virus_pos, new_target)
            tail = dfs(new_pos)
            if tail is None:
                removed_hallways.remove((gate, node))
                continue
            return [(gate, node)] + tail

    answer = dfs(virus_pos)
    return answer if answer else []



def solve(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Решение задачи об изоляции вируса

    Args:
        edges: список коридоров в формате (узел1, узел2)

    Returns:
        список отключаемых коридоров в формате "Шлюз-узел"
    """
    global gateway_hallways, gateways, graph

    edges = list(map(lambda edge: (edge[1], edge[0]) if edge[1].isupper() else edge, edges))
    gateway_hallways = list(
        sorted(filter(lambda edge: edge[0].isupper(), edges))
    )
    graph = _build_graph(edges)
    gateways = [x for x in graph if x.isupper()]
    return _get_disconnected_hallways()


def main():
    edges = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            node1, sep, node2 = line.partition('-')
            if sep:
                edges.append((node1, node2))

    result = solve(edges)
    for edge in result:
        print(f"{edge[0]}-{edge[1]}")


if __name__ == "__main__":
    main()