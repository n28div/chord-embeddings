import numpy as np
import itertools

def create_adjacency_matrix(seq, patterns):
  N = len(seq)
  adj = np.zeros((N, N))

  # compute connections
  for (s1, e1), (s2, e2), score in patterns:
    adj[s1:e1, s2:e2] += score
    adj[s2:e2, s1:e1] += score
  return adj

def salami_annotation_to_vector(sample):
  # assign a number (its index) to each section and repeat that number for the constituents of that
  # category
  annotated_seg = np.array(list(itertools.chain(*[itertools.repeat(name, len(content)) 
                                                for name, _, _, content in sample["structure"]])))
  _, annotated_seg = np.unique(annotated_seg, return_inverse=True)
  return annotated_seg