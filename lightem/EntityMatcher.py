from sklearn.cluster import DBSCAN
import numpy as np
from typing import *
from .Table import Table, TableManager, DATABASE_TEXT_COLUMN_NAME
from .Embedder import Embedder, EmbedderTypes
from .Matcher import Matcher, MatcherTypes
from .Clusterer import Clusterer
import time
import torch

def log(*args):
  now = time.localtime()
  # formata como HH:MM:SS:ms
  formatted_time = time.strftime("%H:%M:%S", now) + f".{int(now.tm_sec * 1000)}"
  print(f"[{formatted_time}]: "+ " ".join(map(str, args)))  

class EntityMatcher():
  def __init__(self, databasePath: str, columnsToText: Dict[str, int], embedderType: EmbedderTypes='glove', gloveModelPath='', sentenceBertDevice='', matcherType: MatcherTypes='cosine', threshold: float=0.9, runInLowMemory: bool=False, batchSize: int=1000, minLenghtToSubCluster: int=5, filterOversizedClusters: int=-1):
    self.databasePath = databasePath
    self.embedderType = embedderType
    self.matcherType = matcherType
    self.threshold = threshold
    self.runInLowMemory = runInLowMemory
    self.batchSize = batchSize
    self.columnsToText = []
    self.minLenghtToSubCluster = minLenghtToSubCluster
    self.filterOversizedClusters = filterOversizedClusters
    for column in columnsToText:
      self.columnsToText.extend([column] * columnsToText[column])
    self.gloveModelPath = gloveModelPath
    self.sentenceBertDevice = sentenceBertDevice if sentenceBertDevice else ('cuda' if torch.cuda.is_available() else 'cpu')
  
  def configureKnn(self, k_neighbors: int):
    '''Configura o KNN para o matcher. O KNN é utilizado para encontrar os pares de entidades semelhantes.'''
    self.k_neighbors = k_neighbors
  
  def __getSubCluster(self, cluster, pairs):
    # constroi a matrix a partir do grafo pra jogar no dbscan
    graph = {}
    
    for i in cluster:
      if i not in pairs:
        continue
      graph[i] = pairs[i]
    
    nodes = list(graph.keys())
    indexes = {nodes[i]: i for i in range(len(nodes))}
    matrix = np.zeros((len(nodes), len(nodes)))
    
    for node, neighbors in graph.items():
      for neighbor, similarity in neighbors.items():
        i = indexes[node]
        if neighbor not in indexes:
          continue
        j = indexes[neighbor]
        distance = 1 - similarity
        matrix[i, j] = distance
        matrix[j, i] = distance
    
    eps = 1 - self.threshold # 
    dbScan = DBSCAN(eps=eps, min_samples=2, algorithm='kd_tree', metric='manhattan')
    labels = dbScan.fit_predict(matrix)
    
    newClusters = {}
    for i, label in enumerate(labels):
      if label not in newClusters:
        newClusters[label] = []
      newClusters[label].append(nodes[i])
    newClusters = [cluster for cluster in newClusters.values()] # nao filtra por len(cluster) > 1 pois tem que retornar todos os nodos que haviam no cluster, para conseguir diferenciar, caso cluster = [1,2,3,4,5] e newCluster = [3,4,5], ele iria achar que nao teria aumentado a quantidade de clusters
    return newClusters

  def __filterOversizedClusters(self):
    if self.filterOversizedClusters == -1:
      return
    self.clusters = [cluster for cluster in self.clusters if len(cluster) <= self.filterOversizedClusters]

  def pipeline(self):
    log("Starting pipeline...")
    self.singleTable: Table = TableManager.createSingleTable(TableManager.openDatabase(self.databasePath))
    log(f"Single table created with {len(self.singleTable.database)} rows.")
    self.singleTable.createTextColumn(self.columnsToText)
    log("Text column created. Creating embedder...")
    self.embedder = Embedder(self.embedderType, self.gloveModelPath)
    embeddings = [self.embedder.getEmbeddings(text) for text in self.singleTable[DATABASE_TEXT_COLUMN_NAME]]
    del self.embedder
    embeddings = [np.array(embedding) for embedding in embeddings]
    log("Embeddings created. Creating matcher...")
    self.matcher = Matcher(embeddings, runInLowMemory=self.runInLowMemory, batchSize=self.batchSize)
    if self.matcherType == 'knn':
      self.matcher.configureKnn(self.k_neighbors)
    self.pairs = self.matcher.getPairs(self.threshold, self.matcherType)
    del self.matcher
    log("Pairs created. Creating graph...")
    self.clusterer = Clusterer()
    self.clusterer.createGraph(self.pairs)
    log("Graph created. Creating clusters...")
    self.clusters = self.clusterer.getClusters()
    del self.clusterer
    # filtra clusters com mais de 1 elemento
    self.clusters = [cluster for cluster in self.clusters if len(cluster) > 1]
    # Post-processing:
    # Deve realizar um dbscan para separar os clusters com transitividade
    pairs = {}
    for i, j, similarity in self.pairs:
      if i not in pairs:
        pairs[i] = {}
      pairs[i][j] = similarity
    self.pairs = pairs
    # pra cada cluster vai fazer o dbscan
    # se achar mais de uma label
    # remove o cluster e adiciona os novos clusters
    for cluster in self.clusters:
      if len(cluster) < self.minLenghtToSubCluster:
        continue
      subCluster = self.__getSubCluster(cluster, self.pairs)
      if len(subCluster) > 1:
        self.clusters.remove(cluster)
        self.clusters.extend([cluster for cluster in subCluster if len(cluster) > 1])
    
    self.__filterOversizedClusters()
    self.clusters = [tuple(cluster) for cluster in self.clusters]
    log("Clusters created.")
    return self.clusters
    