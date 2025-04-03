import numpy as np
from typing import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

MatcherTypes = Literal['cosine', 'euclidean', 'manhattan']

class Matcher:
  '''Classe responsável por comparar embeddings e retornar os pares dado um threshold. Pode ser comparado como maior ou menor que, 
  dependendo da métrica utilizada.'''
  def __init__(self, embeddings: np.ndarray, runInLowMemory:bool=False):
    self.embeddings = embeddings
    self.similarityCheckers = [cosine_similarity]
    self.distanceCheckers = [euclidean_distances, manhattan_distances]
    self.matrixes = []
    self.runLowMemory = runInLowMemory
  
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

  def __runInBatch(self, threshold: float, getPairsFunc: Callable[[float, np.ndarray, np.ndarray], List[Tuple[int, int, float]]]) -> List[Tuple[int, int, float]]:
    '''Executa a função getPairsFunc em batch, dividindo a matriz de embeddings em partes menores para evitar estouro de memória.'''
    print("Running in batch")
    pairs = []
    num_embeddings = len(self.embeddings)
    batch_size = 1000
    for i in range(0, num_embeddings, batch_size):
      batch_embedds = self.embeddings[i:i+batch_size]
      batch_pairs = getPairsFunc(threshold, batch_embedds, self.embeddings)
      
      # Ajusta os índices dos pares do batch para o índice global
      adjusted_pairs = [(p[0] + i, p[1], p[2]) for p in batch_pairs]
      pairs.extend(adjusted_pairs)
      
    return pairs

  def getPairs(self, threshold: float, by: MatcherTypes='cosine') -> List[Tuple[int, int, float]]:
    '''Retorna os pares de instâncias que possuem uma similaridade maior que o threshold. os médotos dispiníveis são:
    - cosine: Similaridade de cosseno: Quanto mais próximo de 1, mais similar. Utilizado de padrão.
    - euclidean: Distância euclidiana: Quanto mais próximo de 0, mais similar.
    - manhattan: Distância de manhattan: Quanto mais próximo de 0, mais similar.
    '''
    if self.runLowMemory:
      match (by):
        case 'cosine':
          return self.__runInBatch(threshold, self.__getPairsCosine)
        case 'euclidean':
          return self.__runInBatch(threshold, self.__getPairsEuclidean)
        case 'manhattan':
          return self.__runInBatch(threshold, self.__getPairsManhattan)
        case _:
          raise Exception(f"Invalid method. Use one of the following: {', '.join([method for method in MatcherTypes.__args__])}.")
    
    if by == 'cosine':
      return self.__getPairsCosine(threshold, self.embeddings, self.embeddings)
    elif by == 'euclidean':
      return self.__getPairsEuclidean(threshold, self.embeddings, self.embeddings)
    elif by == 'manhattan':
      return self.__getPairsManhattan(threshold, self.embeddings, self.embeddings)
    else:
      # puxa os metodos validos de MatcherTypes e salva em uma string
      raise Exception(f"Invalid method. Use one of the following: {', '.join([method for method in MatcherTypes.__args__])}.")
    