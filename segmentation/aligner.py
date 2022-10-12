from typing import List, Dict, Callable
import numpy as np
from functools import partial
from suffix_tree import Tree
import itertools
from collections import deque, namedtuple

def score_function(seq1, seq2, scores: Dict):
  assert "match" in scores, "Scores needs to define a match value"
  assert "mismatch" in scores, "Scores needs to define a mismatch value"
  assert "hole" in scores, "Scores needs to define an hole value"

  assert seq1 is not None or seq2 is not None

  if seq1 == seq2:
    return scores["match"]
  elif seq1 != seq2:
    return scores["mismatch"]
  elif seq1 is None or seq2 is None:
    return scores["hole"]

def local_suffix_alignment(seq: List, score_fn: Callable):
  if score_fn is None:
    score_fn = mk_score_fn()

  # left pad both sequences to allow empty sequences
  s = [None] + seq
  
  n = len(s)
  
  M = np.zeros((n, n))
  traceback = np.full((n, n, 2), -1)

  for i in range(1, n):
    for j in range(i + 1, n):
      v = np.array([
        0,
        M[i - 1, j - 1] + score_fn(s[i], s[j]),
        M[i - 1, j] + score_fn(s[i], None),
        M[i, j - 1] + score_fn(None, s[j])
      ])
      v_traceback = [[0, 0], [i - 1, j - 1], [i - 1, j], [i, j - 1]]

      max_idx = v.argmax()
      M[i, j] = v[max_idx]
      traceback[i, j] = v_traceback[max_idx]

  return M, traceback

def smith_waterman(seq: List, threshold: float = 0, scores: dict = None):
  if callable(scores):
    score_fn = scores
  else:
    if scores is None:
      scores = {"match": 1, "mismatch": -1, "hole": -1}
    score_fn = partial(score_function, scores=scores)
  
  M, traceback = local_suffix_alignment(seq, score_fn)

  # normalize M in range [0, 1]
  M = M / M.max()

  idxs = list(zip(*np.where(M >= threshold)))

  sequences = list()
  for idx in idxs:
    initial_idx = tuple(idx)
    score = M[initial_idx]

    idx = initial_idx
    while M[idx] != 0:
      idx = tuple(traceback[idx])

    i_initial, j_initial = initial_idx
    i_final, j_final = idx
    sequences.append(((i_final, i_initial), (j_final, j_initial), score))
  
  return sequences

def edit_distance(seq1, seq2, v_0_0 = None, c_0 = None, r_0 = None):
  n = len(seq1)
  m = len(seq2)

  M = np.zeros((n + 1, m + 1))

  M[0, 0] = 0 if v_0_0 is None else v_0_0
  M[:, 0] = np.arange(n + 1) if c_0 is None else c_0
  M[0, :] = np.arange(m + 1) if r_0 is None else r_0

  for i in range(1, n + 1):
    for j in range(1, m + 1):
      t_ij = 1 if seq1[i - 1] != seq2[j - 1] else 0
      M[i, j] = min(
        M[i - 1, j] + 1,
        M[i, j - 1] + 1,
        M[i - 1, j - 1] + t_ij
      )
  return M

def needleman_wunsch(seq1, seq2, match = None, mismatch = -1, c_0_0 = None, c_i = None, c_j = None):
  D = mismatch
  M = match
  if M is None or not callable(M):
    M = lambda _1, _2: 1
  
  s1 = [None] + seq1
  s2 = [None] + seq2
  n = len(s1)
  m = len(s2)
  
  C = np.zeros((n, m))
  C[0, 0] = 0 if c_0_0 is None else c_0_0
  C[:, 0] = np.arange(n) * D if c_i is None else c_i
  C[0, :] = np.arange(m) * D if c_j is None else c_j

  for i in range(1, n):
    for j in range(i + 1, m):
      C[i, j] = max(
        C[i - 1, j] + D,
        C[i, j - 1] + D,
        C[i - 1, j - 1] + M(s1[i], s2[j]))
    
  return C

def all_against_all(seq, threshold = 0.8):
  trie = Tree ({ '' : seq })

  trie.root.compute_C ()
  trie.root.compute_left_diverse ()

  nodes = sorted(trie.get_nodes(), key=lambda n: n.string_depth())
  pairs = list(itertools.combinations(nodes, 2))

  subtables = dict()
  pair_key = lambda u, v: f"{id(u)}_{id(v)}"
  subsequences = list()

  for i, (u, v) in enumerate(pairs):
    alpha = u.path.seq
    beta = v.path.seq

    up_vp = pair_key(u.parent, v.parent)
    u_vp = pair_key(u, v.parent)
    up_v = pair_key(u.parent, v)
    u_v = pair_key(u, v)
    
    v_0_0 = subtables[up_vp][-1, -1] if up_vp in subtables else None
    c_0 = subtables[u_vp][:, -1] if u_vp in subtables else None
    r_0 = subtables[up_v][-1, :] if up_v in subtables else None
    subtables[u_v] = edit_distance(alpha, beta, v_0_0=v_0_0, c_0=c_0, r_0=r_0)
    
    # check for values above threshold in subtable
    #u_len, v_len = map(tuple, np.where(subtables[u_v] < 2))

    #u_starts = list()
    #u.pre_order(lambda n: u_starts.append(n.path.start) if n.is_leaf() else None)

    #v_starts = list()
    #v.pre_order(lambda n: v_starts.append(n.path.start) if n.is_leaf() else None)

    #subsequences.extend([
    #  ((su, su + lu), (sv, sv + lv))
    #  for (su, lu), (sv, lv) in zip(
    #    itertools.product(u_starts, u_len), 
    #    itertools.product(v_starts, v_len))
    #  if lu > 0 and lv > 0])
  return subsequences

def bitap(seq, threshold, match = lambda _1, _2: 1, mismatch = -1):
  #https://ieeexplore.ieee.org/document/796573
  active_set = deque()
  subsets = list()

  trie = Tree({"": seq})

  Elem = namedtuple("Elem", ["n1", "n2", "c_0_0", "c_i", "c_j"])
  active_set.append(Elem(trie.root, trie.root, None, None, None))
  while len(active_set) > 0:
    n1, n2, c_0_0, c_i, c_j = active_set.pop()

    n1_subtrees = list()
    n2_subtrees = list()
    n1.pre_order(lambda n: n1_subtrees.append(n))
    n2.pre_order(lambda n: n2_subtrees.append(n))

    # start from 1 to skip root node which doenst represent any symbol
    for i in range(1, len(n1_subtrees)):
      subtree_1 = n1_subtrees[i]
      t1 = list(subtree_1.path.S[subtree_1.path.start:subtree_1.path.end])

      for j in range(i, len(n2_subtrees)):
        subtree_2 = n2_subtrees[j]
        t2 = list(subtree_2.path.S[subtree_2.path.start:subtree_2.path.end])

        c = needleman_wunsch(t1, t2, 
          match=match, 
          mismatch=mismatch, 
          c_0_0=c_0_0, 
          c_i=c_i, 
          c_j=c_j)
        


        pass
    
    pass

    #for n1_idx, n1_subtrees in enumerate(n1_subtrees):
    #  for n2_idx in range(n1_idx, len(n2_subtrees)):
    #for sub1, sub2 in itertools.combinations(n1, n2, 2):
    #  pass


  



if __name__ == "__main__":
  from salami_choco_data import Dataset
  data = Dataset(salami_path="/home/n28div/university/thesis/salami-data-public/",
               choco_path="/home/n28div/university/thesis/choco/")
  sample = next(data.iter())

  ranges = smith_waterman(sample["chords"], threshold=0.7)
  #subseqs = all_against_all(sample["chords"])
  print(c)