import sys

gateway_hallways: list[tuple[str, str]]

def solve(edges: list[tuple[str, str]]) -> list[str]:
    """
    Решение задачи об изоляции вируса

    Args:
        edges: список коридоров в формате (узел1, узел2)

    Returns:
        список отключаемых коридоров в формате "Шлюз-узел"
    """

    result = []
    edges = list(map(lambda edge: (edge[1], edge[0]) if edge[1].isupper() else edge, edges))
    gateway_hallways = list(
        sorted(filter(lambda edge: edge[0].isupper(), edges))
    )
    
    print(gateway_hallways)
    return result


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
        print(edge)


if __name__ == "__main__":
    main()