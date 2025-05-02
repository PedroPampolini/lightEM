from typing import *

class Clusterer():
  '''Classe responsável por agrupar nós em clusters. Um cluster é uma lista de nós que estão conectados entre si.'''
  def __init__(self):
    self.graph: Dict[int, Set[Tuple[int, float]]] = {}
    self.clusters: List[Tuple[int]] = []
  
  def createGraph(self, pairs: List[Tuple[int, int, float]]) -> None:
    '''Cria um grafo a partir de uma lista de pares. Cada par representa uma aresta.'''
    self.edges = pairs
    print(f"Creating graph with {len(self.edges)} edges...")
    nodeIds = set([self.edge[0] for self.edge in self.edges] + [self.edge[1] for self.edge in self.edges])
    print(f"Found {len(nodeIds)} nodes.")
    self.graph = {nodeId: set() for nodeId in nodeIds}
    edgeIndex = 0
    edgeCount = len(self.edges)
    print(f"Adding edges to graph...")
    for edge in self.edges:
      self.graph[edge[0]].add((edge[1], edge[2]))
      self.graph[edge[1]].add((edge[0], edge[2]))
      edgeIndex += 1
      porcentagem = (edgeIndex / edgeCount) * 100
      if edgeIndex % 100 == 0:
        print(f"\r[{int(porcentagem // 2) * '='}{int((100 - porcentagem) // 2) * '-'}] {porcentagem:.2f}%", end='')
    print()
  
  def __exploreNode(self, start_node: int, explored: Set[int], unexplored: Set[int]) -> List[int]:
    '''Explora um nó iterativamente e retorna todos os nós explorados a partir dele.'''
    stack = [start_node]
    cluster = []

    while stack:
      node = stack.pop()
      if node not in explored:
        explored.add(node)
        unexplored.discard(node)
        cluster.append(node)
        stack.extend(neighbor[0] for neighbor in self.graph[node] if neighbor not in explored)

    return cluster

  def getClusters(self) -> List[Tuple[int]]:
    '''Retorna os clusters do grafo. Um cluster é uma lista de nós que estão conectados entre si.'''
    unexplored = set(self.graph.keys())
    explored = set()
    clusters = []

    while unexplored:
      node = next(iter(unexplored))  # Obtém um elemento arbitrário do conjunto
      cluster = self.__exploreNode(node, explored, unexplored)
      clusters.append(tuple(cluster))
  
    return clusters
    