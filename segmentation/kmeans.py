from typing import List, Dict, Iterable
import numpy as np
from sklearn.cluster import KMeans
from segmentation.pattern_extraction import extract_patterns
from segmentation.utils import create_adjacency_matrix

def segment_kmeans(X, threshold: float = 0.9, scores: Dict = None, n_clusters: int = 8, random_seed: int = 42):
  if not(isinstance(X, Iterable)):
    X = [X]
  
  X_patterns = [extract_patterns(seq, threshold=threshold, scores=scores) for seq in X]
  connections = [create_adjacency_matrix(seq, patterns) for seq, patterns in zip(X, X_patterns)]
  kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed).fit(connections)
  _, pred_seg = np.unique(kmeans.labels_, return_inverse=True)
  return pred_seg