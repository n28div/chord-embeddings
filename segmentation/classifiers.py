from typing import Dict, List
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np

class SmithWatermanSimilarityVectorizer(TransformerMixin, BaseEstimator):
  def __init__(self, threshold: float = 0.0, score: Dict = {"match": 1, "mismatch": -1, "hole": -1}, full: bool = False):
    self.threshold = threshold
    self.score = score
    self.full = full

  def fit(self, X, y = None):
    self._cached_patterns = dict()
    return self

  def transform(self, X):
    X_id = sequence_id(X)

    if X_id in self._cached_patterns:
      adj = self._cached_patterns[X_id]
    else:
      patterns = smith_waterman(X, threshold=self.threshold, scores=self.score)
      # remove duplicate elements from ranges
      patterns = set(patterns)
      # filter out identity ranges (i.e. ((0, 10), (0, 10), score))
      patterns = filter(lambda r: r[0] != r[1], patterns)
      patterns = list(patterns)

      # one dimension for each pattern, each row of a pattern
      # represent an element in the sequence and each column
      # other character
      # M takes value 1 if an element can be aligned to another
      # element in some way
      # the relation is symmetrical
      M = np.zeros((len(patterns), len(X), len(X)))

      for i, ((s1, e1), (s2, e2), score) in enumerate(patterns):
        M[i, s1:e1, s2:e2] = 1
        M[i, s2:e2, s1:e1] = 1
      
      if not self.full:
        M = M.sum(axis=0)

      self._cached_patterns[X_id] = M

    return M
