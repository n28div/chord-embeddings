import numpy as np
from sklearn.metrics import make_scorer
import itertools

def pairwise_metrics(annotated, predicted):
  # from Structural Segmentation of Musical Audio by Constrained Clustering, Levy (2008)
  def _compute_pairwise(segm):
    labels = np.unique(segm)
    return set(itertools.chain(*[itertools.product(np.where(segm == label)[0], repeat=2) for label in labels]))

  M_ann = _compute_pairwise(annotated)
  M_pred = _compute_pairwise(predicted)

  inters = len(M_pred.intersection(M_ann))
  precision =  inters / (len(M_pred) + 1e-50)
  recall = inters / (len(M_ann) + 1e-50)
  f1 = (2 * precision * recall) / (precision + recall + 1e-50)

  return precision, recall, f1


def under_over_segmentation(annotated, predicted, acc: str = None):
  # evaluate according to Lukashevich (IMIR, 2008)
  N = len(annotated)
  positions_A = [(label, np.where(annotated == label)[0]) for label in np.unique(annotated)]
  positions_P = [(label, np.where(predicted == label)[0]) for label in np.unique(predicted)]

  N_A = len(positions_A)
  N_P = len(positions_P)

  # compute joint probability
  n = np.zeros((N_A, N_P))
  for i, i_elem in positions_A:
    for j, j_elem in positions_P:
      n[i, j] = len(np.intersect1d(i_elem, j_elem))
  p = n / n.sum()

  # compute single probabilities
  n_A = np.array([len(elem) for _, elem in positions_A])
  p_A = n_A / n.sum()
  n_P = np.array([len(elem) for _, elem in positions_P])
  p_P = n_P / n.sum()

  # compute entropies
  h_EA = 0
  for i, _ in positions_A:
    partial = 0
    for j, _ in positions_P:
      p_ji_EA = n[i, j] / n_A[i]
      partial += p_ji_EA * np.log2(p_ji_EA + 1e-50)
    h_EA += p_A[i] * partial
  h_EA = - h_EA

  h_AE = 0
  for j, _ in positions_P:
    partial = 0
    for i, _ in positions_A:
      p_ij_AE = n[i, j] / n_P[j]
      partial += p_ij_AE * np.log2(p_ij_AE + 1e-50)
    h_AE += p_P[j] * partial
  h_AE = -h_AE

  over_segmentation = max(0, 1 - (h_EA / (np.log2(N_P + 1e-50) + 1e-50)))
  under_segmentation = max(0, 1 - (h_AE / (np.log2(N_A + 1e-50) + 1e-50)))

  if acc == "sum":
    score = over_segmentation + under_segmentation
  elif acc == "mean":
    score = (over_segmentation + under_segmentation) / 2
  else:
    score = (over_segmentation, under_segmentation)
  
  return score