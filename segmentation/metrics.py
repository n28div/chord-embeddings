import numpy as np
from sklearn.metrics import confusion_matrix
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


def under_over_segmentation(annotated, predicted, smoothing_alpha: float  = 1e-100):
  # evaluate according to Lukashevich (IMIR, 2008)
  L_a = len(np.unique(annotated))
  L_e = len(np.unique(predicted))
  F = len(annotated)
  cm = confusion_matrix(annotated, predicted)

  # compute joint and marginal distributions
  P = (cm + smoothing_alpha) / (F + (L_a * L_e * smoothing_alpha))
  P_a = P.sum(axis=0) / F
  P_e = P.sum(axis=1) / F

  # compute conditional distributions
  P_ae = (cm + smoothing_alpha) / (cm.sum(axis=1) + (L_a * smoothing_alpha))
  P_ea = (cm + smoothing_alpha) / (cm.sum(axis=0) + (L_e * smoothing_alpha))

  # compute conditional entropies
  H_ea = -1 * (P_a * (P_ea * np.log2(P_ea)).sum(axis=0)).sum()
  H_ae = -1 * (P_e * (P_ae * np.log2(P_ae)).sum(axis=1)).sum()

  over_segmentation = 1 - (H_ea / np.log2(L_e)).clip(0, 1)
  under_segmentation = 1 - (H_ae / np.log2(L_a)).clip(0, 1)
  
  return under_segmentation, over_segmentation