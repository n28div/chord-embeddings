from typing import List, Dict
from segmentation.aligner import smith_waterman

def extract_patterns(seq, threshold: float = 0.9, scores: Dict = None) -> List:
  ranges = smith_waterman(seq, threshold=threshold, scores=scores)
  # remove duplicate elements from ranges
  ranges = set(ranges)
  # filter out identity ranges (i.e. ((0, 10), (0, 10), score))
  ranges = filter(lambda r: r[0] != r[1], ranges)

  return list(ranges)