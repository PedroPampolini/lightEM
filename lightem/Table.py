import pandas as pd
import numpy as np
import os
from typing import *

DATABASE_TEXT_COLUMN_NAME = "matchingText"

class Table():
  '''Classe responsável por gerenciar uma tabela de dados. Criação de colunas e cálculo de embeddings.'''
  def __init__(self, database: pd.DataFrame):
    self.database = database
    self.embeddings = None
    
  def createTextColumn(self, textColumns: List[str]) -> None:
    '''Cria uma coluna que representa o texto a ser utilizado para o embedding e consequentemente
    para o matching. A coluna é criada a partir da concatenação das colunas passadas como argumento.'''
    textColumnNotWeighted = list(set(textColumns))
    self.__checkColumnsExist(textColumnNotWeighted)
    self.database[textColumnNotWeighted] = self.database[textColumnNotWeighted].astype(str)
    self.database[textColumnNotWeighted] = self.database[textColumnNotWeighted].replace(np.nan, '', regex=True)
    self.database[DATABASE_TEXT_COLUMN_NAME] = self.database[textColumns].apply(lambda row: " ".join(row), axis=1)
    
  def __checkColumnsExist(self, columns: List[str]) -> None:
    '''Checa se as colunas passadas como argumento existem na tabela. Se alguma coluna não existir, uma exceção é lançada.'''
    for column in columns:
      if column not in self.database.columns:
        raise Exception(f"Column {column} does not exist in the database. Existing columns are {self.database.columns}")
  
  def __getitem__(self, key: str) -> pd.Series:
    '''Retorna uma coluna da tabela.'''
    return self.database[key]
  
  # recebe uma função que recebe uma string e retorna um vetor de embeddings
  def getEmbeddings(self, embedderFunction: Callable[[str], np.ndarray]) -> List[np.ndarray]:
    '''Calcula e retorna os embeddings de cada linha da tabela.'''
    # self.embeddings = self.database[DATABASE_TEXT_COLUMN_NAME].apply(embedderFunction)
    self.embeddings = [embedderFunction(text) for text in self.database[DATABASE_TEXT_COLUMN_NAME]]
    return self.embeddings

class TableManager():
  '''Classe responsável por gerenciar tabelas de dados. Como abrir os bancos dado um path, já tratando possíveis erros,
  criar uma tabela unificada a partir de várias tabelas.'''
  def __init__(self):
    pass
  
  def openDatabase(path: str):
    # checa se o path existe
    if not os.path.exists(path):
      raise Exception(f"Path {path} does not exist. Current path is {os.getcwd()}")
    
    # checa se o path é um arquivo
    if os.path.isfile(path):
      return [pd.read_csv(path)]
    
    # checa se o path é um diretório
    if os.path.isdir(path):
      databases = []
      for file in os.listdir(path):
        if file.endswith(".csv"):
          databases.append(pd.read_csv(os.path.join(path, file)))
        elif file.endswith(".xlsx"):
          databases.append(pd.read_excel(os.path.join(path, file)))
      return databases
  
  def createSingleTable(databases: List[pd.DataFrame]) -> Table:
    '''
    Cria uma tabela unificada a partir de várias tabelas. 3 Colunas são adicionadas para um controle de origem de cada
    instância:
    - databaseIndex: Index da tabela original
    - rowIndex: Index da linha na tabela original.
    - globalIndex: Index global da linha
    '''
    TableManager.checkNameColumns(databases)
    rowIndex = 0
    for dbIndex, db in enumerate(databases):
      db["databaseIndex"] = dbIndex
      db["rowIndex"] = np.arange(0, db.shape[0])
      db['globalIndex'] = np.arange(rowIndex, rowIndex + db.shape[0])
      rowIndex += db.shape[0]
    database = pd.concat(databases, ignore_index=True)
    return Table(database)
  
  def checkNameColumns(databases: List[pd.DataFrame]):
    '''Checa se os nomes databaseIndex, rowIndex e globalIndex já existem nas tabelas. Se existirem, uma exceção é lançada
    para renomear as colunas para evitar conflitos.'''
    for db in databases:
      if 'databaseIndex' in db.columns or 'rowIndex' in db.columns or 'globalIndex' in db.columns:
        raise Exception("Columns databaseIndex, rowIndex and globalIndex are reserved. Please rename them.")
  