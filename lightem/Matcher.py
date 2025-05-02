import numpy as np
import math
from typing import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

MatcherTypes = Literal['cosine', 'euclidean', 'manhattan', 'knn']

class Matcher:
  '''Classe responsável por comparar embeddings e retornar os pares dado um threshold. Pode ser comparado como maior ou menor que, 
  dependendo da métrica utilizada.'''
  def __init__(self, embeddings: np.ndarray, runInLowMemory:bool=False, batchSize:int=1000) -> None:
    self.embeddings = embeddings
    self.similarityCheckers = [cosine_similarity]
    self.distanceCheckers = [euclidean_distances, manhattan_distances]
    self.matrixes = []
    self.runLowMemory = runInLowMemory
    self.batchSize = batchSize
  
  def setEmbeddings(self, embeddings: np.ndarray) -> None:
    '''Define os embeddings a serem utilizados.'''
    self.embeddings = embeddings
    
  def __getPairsCosine(self, threshold: float, embedds1: np.ndarray, embedds2: np.ndarray) -> List[Tuple[int, int, float]]:
    '''Calcula a similaridade de cosseno e retorna os pares que possuem uma similaridade maior ou igual ao threshold'''
    similarityMatrix = cosine_similarity(embedds1, embedds2)
    pairs = np.argwhere(similarityMatrix >= threshold)
    pairs = [(p[0], p[1], similarityMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    return pairs
  
  def __getPairsEuclidean(self, threshold: float, embedds1: np.ndarray, embedds2: np.ndarray) -> List[Tuple[int, int, float]]:
    '''Calcula a distância euclidiana e retorna os pares que possuem uma distância menor ou igual ao threshold'''
    distanceMatrix = euclidean_distances(embedds1, embedds2)
    pairs = np.argwhere(distanceMatrix <= threshold)
    pairs = [(p[0], p[1], distanceMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    return pairs
  
  def __getPairsManhattan(self, threshold: float, embedds1: np.ndarray, embedds2: np.ndarray) -> List[Tuple[int, int, float]]:
    '''Calcula a distância de manhattan e retorna os pares que possuem uma distância menor ou igual ao threshold'''
    distanceMatrix = manhattan_distances(embedds1, embedds2)
    pairs = np.argwhere(distanceMatrix <= threshold)
    pairs = [(p[0], p[1], distanceMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    return pairs

  def __getPairsKNN(self, threshold: float, embedds1: np.ndarray, embedds2: np.ndarray) -> List[Tuple[int, int, float]]:
    # Interpreta o 'threshold' como número de vizinhos k
    k = int(threshold)
    pairs = []  # Lista de tuplas (i, j, distância) a ser retornada

    # Verifica se os dois conjuntos de embeddings são o mesmo (mesma referência) para evitar auto-match
    same_data = embedds1 is embedds2

    # Determina quantos vizinhos buscar:
    # - Se for o mesmo conjunto, busca k+1 vizinhos (um extra para possivelmente ser o próprio ponto).
    # - Se forem conjuntos distintos, busca exatamente k vizinhos.
    n_neighbors = k
    if same_data:
        n_neighbors = min(k + 1, embedds2.shape[0])  # k+1 ou o total de pontos, o que for menor.
    else:
        n_neighbors = min(k, embedds2.shape[0])      # no máximo o tamanho de embedds2.

    # Configura o NearestNeighbors com distância Euclidiana e ajusta no conjunto embedds2
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
    nbrs.fit(embedds2)

    # Encontra os vizinhos mais próximos de cada vetor em embedds1
    distances, indices = nbrs.kneighbors(embedds1)

    # Monta os pares (i, j, dist) de acordo com os vizinhos encontrados
    for i in range(len(embedds1)):
        # Converte resultados para listas mutáveis (facilita remoção, se necessário)
        neighbor_idxs = list(indices[i])
        neighbor_dists = list(distances[i])

        # Se estiver no modo "mesmo conjunto", remove o próprio índice (self) caso apareça nos vizinhos
        if same_data:
            if i in neighbor_idxs:
                idx = neighbor_idxs.index(i)
                # Remove o próprio ponto das listas de índices e distâncias
                neighbor_idxs.pop(idx)
                neighbor_dists.pop(idx)
        # Se ainda houver um vizinho extra (por exemplo, buscou k+1 e o próprio ponto não estava incluído),
        # remove o último para manter apenas k vizinhos no resultado final.
        if len(neighbor_idxs) > k:
            neighbor_idxs = neighbor_idxs[:k]
            neighbor_dists = neighbor_dists[:k]

        # Adiciona os pares (i, j, distância) para cada vizinho válido
        for idx, dist in zip(neighbor_idxs, neighbor_dists):
            pairs.append((i, idx, dist))
    return pairs

  def __runInBatch(self, threshold: float,getPairsFunc: Callable[[float, np.ndarray, np.ndarray], List[Tuple[int, int, float]]]) -> List[Tuple[int, int, float]]:
        """
        Divide a matriz (NxN) em blocos de (BxB), calcula cada sub-matriz
        só uma vez (triângulo superior) e junta todos os pares que passam
        no threshold.

        • B = self.batchSize  
        • Evita duplicatas (mantém apenas i < j).
        """
        B = self.batchSize
        N = len(self.embeddings)

        n_blocks = math.ceil(N / B)
        total_blocks = n_blocks * (n_blocks + 1) // 2
        done = 0

        pairs: List[Tuple[int, int, float]] = []

        for rb in range(n_blocks):
            r_start, r_end = rb * B, min((rb + 1) * B, N)
            rows = self.embeddings[r_start:r_end]

            for cb in range(rb, n_blocks):
                c_start, c_end = cb * B, min((cb + 1) * B, N)
                cols = self.embeddings[c_start:c_end]

                # calcula a sub-matriz (BxB ou menos) de dist./sim.
                block_pairs = getPairsFunc(threshold, rows, cols)

                # ajusta índices locais -> globais e descarta duplicatas
                for i_local, j_local, score in block_pairs:
                    i_global, j_global = r_start + i_local, c_start + j_local
                    if i_global < j_global:
                        pairs.append((i_global, j_global, score))

                done += 1
                pct = done / total_blocks * 100
                print(f"\r[{int(pct//2)*'='}{int((100-pct)//2)*'-'}] {pct:6.2f}%", end='')

        print()
        return pairs

  def configureKnn(self, k: int) -> None:
    self.knn_neighbors = k

  def getPairs(self, threshold: float, by: MatcherTypes='cosine') -> List[Tuple[int, int, float]]:
    '''Retorna os pares de instâncias que possuem uma similaridade maior que o threshold. os médotos dispiníveis são:
    - cosine: Similaridade de cosseno: Quanto mais próximo de 1, mais similar. Utilizado de padrão.
    - euclidean: Distância euclidiana: Quanto mais próximo de 0, mais similar.
    - manhattan: Distância de manhattan: Quanto mais próximo de 0, mais similar.
    '''
    lowMemoryRunner = self.__runInBatch if self.runLowMemory else None
    self.embeddings = np.array(self.embeddings)  if type(self.embeddings) != np.ndarray else self.embeddings
    func = None
    if by == 'cosine':
      func = self.__getPairsCosine
    elif by == 'euclidean':
      func = self.__getPairsEuclidean
    elif by == 'manhattan':
      func = self.__getPairsManhattan
    elif by == 'knn':
      func = self.__getPairsKNN
      threshold = self.knn_neighbors  # Interpreta o threshold como número de vizinhos k
    else:
      raise Exception(f"Invalid method. Use one of the following: {', '.join([method for method in MatcherTypes.__args__])}.")
    
    if self.runLowMemory:
      return lowMemoryRunner(threshold, func)
    else:
      return func(threshold, self.embeddings, self.embeddings)
    
    # if self.runLowMemory:
    #   match (by):
    #     case 'cosine':
    #       return self.__runInBatch(threshold, self.__getPairsCosine)
    #     case 'euclidean':
    #       return self.__runInBatch(threshold, self.__getPairsEuclidean)
    #     case 'manhattan':
    #       return self.__runInBatch(threshold, self.__getPairsManhattan)
    #     case _:
    #       raise Exception(f"Invalid method. Use one of the following: {', '.join([method for method in MatcherTypes.__args__])}.")
    
    # if by == 'cosine':
    #   return self.__getPairsCosine(threshold, self.embeddings, self.embeddings)
    # elif by == 'euclidean':
    #   return self.__getPairsEuclidean(threshold, self.embeddings, self.embeddings)
    # elif by == 'manhattan':
    #   return self.__getPairsManhattan(threshold, self.embeddings, self.embeddings)
    # else:
    #   # puxa os metodos validos de MatcherTypes e salva em uma string
    #   raise Exception(f"Invalid method. Use one of the following: {', '.join([method for method in MatcherTypes.__args__])}.")
    