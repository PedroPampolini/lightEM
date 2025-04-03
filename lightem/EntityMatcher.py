from sklearn.cluster import DBSCAN
import numpy as np
from typing import *
from .Table import Table, TableManager, DATABASE_TEXT_COLUMN_NAME
from .Embedder import Embedder, EmbedderTypes
from .Matcher import Matcher, MatcherTypes
from .Clusterer import Clusterer

class EntityMatcher():
  def __init__(self, databasePath: str, columnsToText: Dict[str, int], embedderType: EmbedderTypes='glove', gloveModelPath='', matcherType: MatcherTypes='cosine', threshold: float=0.9, runInLowMemory: bool=False):
    self.databasePath = databasePath
    self.embedderType = embedderType
    self.matcherType = matcherType
    self.threshold = threshold
    self.runInLowMemory = runInLowMemory
    self.columnsToText = []
    for column in columnsToText:
      self.columnsToText.extend([column] * columnsToText[column])
    self.gloveModelPath = gloveModelPath
      
  
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
    
    eps = 1 - self.threshold
    dbScan = DBSCAN(eps=eps, min_samples=2, algorithm='kd_tree', metric='manhattan')
    labels = dbScan.fit_predict(matrix)
    
    newClusters = {}
    for i, label in enumerate(labels):
      if label not in newClusters:
        newClusters[label] = []
      newClusters[label].append(nodes[i])
    newClusters = [cluster for cluster in newClusters.values()] # nao filtra por len(cluster) > 1 pois tem que retornar todos os nodos que haviam no cluster, para conseguir diferenciar, caso cluster = [1,2,3,4,5] e newCluster = [3,4,5], ele iria achar que nao teria aumentado a quantidade de clusters
    return newClusters

  def pipeline(self):
    print("Starting pipeline...")
    self.singleTable: Table = TableManager.createSingleTable(TableManager.openDatabase(self.databasePath))
    print(f"Single table created with {len(self.singleTable.database)} rows.")
    self.singleTable.createTextColumn(self.columnsToText)
    print("Text column created.")
    self.embedder = Embedder(self.embedderType, self.gloveModelPath)
    embeddings = [self.embedder.getEmbeddings(text) for text in self.singleTable[DATABASE_TEXT_COLUMN_NAME]]
    embeddings = [np.array(embedding) for embedding in embeddings]
    print("Embeddings created.")
    self.matcher = Matcher(embeddings, runInLowMemory=self.runInLowMemory)
    if self.matcherType == 'knn':
      self.matcher.configureKNN(self.k_neighbors, self.seed, self.metric, self.dim)
    self.pairs = self.matcher.getPairs(self.threshold, self.matcherType)
    print("Pairs created.")
    self.clusterer = Clusterer()
    self.clusterer.createGraph(self.pairs)
    print("Graph created. Creating clusters...")
    self.clusters = self.clusterer.getClusters()
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
      subCluster = self.__getSubCluster(cluster, self.pairs)
      if len(subCluster) > 1:
        self.clusters.remove(cluster)
        self.clusters.extend([cluster for cluster in subCluster if len(cluster) > 1])
    print("Clusters created.")
    return self.clusters
    