from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import *


EmbedderTypes = Literal['glove', 'sentence-transformers']

class Embedder():
  '''Classe responsável por calcular embeddings de uma sentença. Alguns embedders são aceitos, sendo eles:
  - sentence-transformers: Mais lento mas com embeddings de melhor qualidade
  - glove: Mais rápido mas com embeddings de qualidade inferior'''
  
  def __init__(self, embedderType: EmbedderTypes):
    self.embedderType = embedderType
    match embedderType:
      case 'glove':
        self.embedder = KeyedVectors.load_word2vec_format("models/bin/glove.6B.300d.bin", binary=True)
      case 'sentence-transformers':
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
      
  
  def __preprocessSentence(self, sentence: str) -> str:
    '''Remove stopwords, caracteres da sentença e deixa tudo em minúsculo.'''
    specialChars = ['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'"]
    for char in specialChars:
      sentence = sentence.replace(char, ' ')
    
    return sentence.lower()

  def preprocess(self, sentence):
    return self.__preprocessSentence(sentence)
  
  def getEmbeddings(self, sentence: str) -> np.ndarray:
    '''Calcula o embedding de uma única sentença. Se removeStopwords for True, as stopwords são removidas da sentença.
    É calculado o embedding de cada palavra e, para retornar um embedding único, é feita a média dos embeddings.
    OBS.: Verificar se retornar a média é a melhor forma de retornar um embedding único. Pensar em como dar destaque para
    palavras mais importantes.'''
    sentence = self.__preprocessSentence(sentence)
    match self.embedderType:
      case 'glove':
        return self.__getEmbeddingsGlove(sentence)
      case 'sentence-transformers':
        return self.__getEmbeddingsSentenceTransformers(sentence)

  def __getEmbeddingsSentenceTransformers(self, sentence) -> np.ndarray:
    '''Retorna os embeddings de uma sentença utilizando o modelo de Sentence-Transformers all-MiniLM-L12-v2. Mais robusto e com melhores resultados, mas mais lento.'''
    return self.embedder.encode(sentence)

  def __getEmbeddingsGlove(self, sentence) -> List[np.ndarray]:
    '''Retorna os embeddings de uma sentença utilizando o modelo Glove. Se uma palavra não estiver no vocabulário, ela é ignorada.'''
    words = sentence.split()
    embeddings = [self.embedder[word] for word in words if word in self.embedder.key_to_index]
    if not embeddings:
      return np.zeros(self.embedder.vector_size)
    return np.mean(embeddings, axis=0)  # Otimizar a função de agrupamento de embeddings, média seria o suficiente/certo?
  