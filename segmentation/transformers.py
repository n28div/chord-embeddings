from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from segmentation.aligner import smith_waterman
import numpy as np
from suffix_tree import Tree

def sequence_id(seq: List[str]) -> str:
    # check if X has previously been cached in the _cached_patterns
    # hashmap: hashes are X contents joined by the first unicode
    # private use area char U+E000
    return "\ue000".join(seq)

class SmithWatermanSimilarityAdjacencyVectorizer(TransformerMixin, BaseEstimator):
  def __init__(self, threshold: float = 0.85, 
                     score: Dict = {"match": 1, "mismatch": -1, "hole": -1}, 
                     weighted_edges: bool = False):
    self.threshold = threshold
    self.score = score
    self.weighted_edges = weighted_edges

  def fit(self, X, y = None):
    self._cached_patterns = dict()
    return self

  def _range_score(self, r1: Tuple[int, int], r2: Tuple[int, int], score: float) -> float:
    if self.weighted_edges:
      return score
    else:
      return 1

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
      M = np.zeros((len(X), len(X)))

      for (s1, e1), (s2, e2), score in patterns:
        M[s1:e1, s2:e2 + 1] += self._range_score((s1, e1), (s2, e2), score)
        M[s2:e2, s1:e1 + 1] += self._range_score((s2, e2), (s1, e1), score)

      self._cached_patterns[X_id] = M

    return M

class SmithWatermanSimilarityVotingVectorizer(TransformerMixin, BaseEstimator):
  def __init__(self, threshold: float = 0.85, 
                     score: Dict = {"match": 1, "mismatch": -1, "hole": -1},
                     weighted_vote = False):
    self.threshold = threshold
    self.score = score
    self.weighted_vote = weighted_vote

  def fit(self, X, y = None):
    self._cached_patterns = dict()
    return self

  def transform(self, X):
    X_id = sequence_id(X)

    if X_id in self._cached_patterns:
      votes = self._cached_patterns[X_id]
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
      votes = np.zeros((len(X), len(patterns)))

      for i, ((s1, e1), (s2, e2), score) in enumerate(patterns):
        vote_weight = 1 if not self.weighted_vote else score
        votes[s1:e1, i] = 1 * vote_weight
        votes[s2:e2, i] = 1 * vote_weight
        
      self._cached_patterns[X_id] = votes

    return votes

class SimilarToMaximalVectorizer(TransformerMixin, BaseEstimator):
  def __init__(self, threshold: float = 0.85, 
                     score: Dict = {"match": 1, "mismatch": -1, "hole": -1}):
    self.threshold = threshold
    self.score = score

  def fit(self, X, y = None):
    self._cached_patterns = dict()
    return self

  def transform(self, X):
    X_id = sequence_id(X)

    if X_id in self._cached_patterns:
      votes = self._cached_patterns[X_id]
    else:
      trie = Tree({"": X})
      maximals = trie.maximal_repeats()
  
      patterns = list()
      for sm in maximals:
        seq = list(sm[1].seq)
        similars = set(s[1] for s in smith_waterman(seq, X, threshold=self.threshold))

        pattern = np.zeros(len(X))
        for start, end in similars:
          pattern[start:end] = 1
        patterns.append()

      self._cached_patterns[X_id] = np.stack(X)

    return self._cached_patterns[X_id]