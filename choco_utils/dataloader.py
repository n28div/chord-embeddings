from typing import Iterable, List, Iterator
import jams
import os
from os import path as osp
import collections
from lark import Lark
from harte2vec.harte import HARTE_LARK_GRAMMAR

ChoCoDocument = collections.namedtuple("ChoCoDocument", ["annotations", "source", "jams"])
HarteAnnotation = collections.namedtuple("HarteAnnotation", ["symbol", "duration"])
HarteSectionAnnotation = collections.namedtuple("HarteSectionAnnotation", ["chord", "section"])

class ChoCoCorpus(object):
  """
  Iterate over all documents contained in ChoCo to retrieve jams files.
  """

  def __init__(self, choco_path: str):
    """
    Args:
        choco_path (str): Path to ChoCo directory.
    """
    partitions_path = osp.join(choco_path, "partitions")
    self.partitions_paths = [osp.join(partitions_path, p, "choco") for p in os.listdir(partitions_path)]

  def _jams_in_partition(self, partition_path: str) -> Iterator[str]:
    """
    Iterate over all the jams file in a partition.
    Automatically taked the converted chords if chords are not in Harte encoding.

    Args:
        partition_path (str): The path to the ChoCo partition.

    Yields:
        Iterator[str]: Path to jams file contained in the partition.
    """
    partition_dirs = os.listdir(partition_path)
    jams_dir = "jams-converted" if "jams-converted" in partition_dirs else "jams"
    jams_path = osp.join(partition_path, jams_dir)
    
    jams_files = os.walk(jams_path)
    for (root, _, files) in jams_files:
      for file in files:
        file_path = osp.join(root, file)
        yield file_path
  
  def __iter__(self) -> Iterator[jams.JAMS]:
    """
    Iterate over all chords progressions.
    Subclass this method to provide custom behaviour.
    By default yields the loaded jams.

    Yields:
        Iterator[jams.JAMS]: Loaded JAMS file.
    """
    for partition_path in self.partitions_paths:
      for jam_path in self._jams_in_partition(partition_path):
        yield jams.load(jam_path, validate=False)


class ChoCoHarteAnnotationsCorpus(ChoCoCorpus):
  """
  Iterate over all documents contained in ChoCo to retrieve chords in Harte format.
  """
  def __iter__(self) -> Iterator[List[ChoCoDocument]]:
    """
    Yields:
        Iterator[List[HarteAnnotation]]: Progression of chords in Harte format.
    """
    for partition_path in self.partitions_paths:
      for jam_path in self._jams_in_partition(partition_path):
        jam = jams.load(jam_path, validate=False)
        namespaces = [ str(a.namespace) for a in jam.annotations ]
        
        chord_namespace = "chord_harte" if "chord_harte" in namespaces else "chord"
        annotation = jam.search(namespace=chord_namespace)
        observations = annotation[0].data
        chords = [HarteAnnotation(obs.value, obs.duration) for obs in observations]
        yield ChoCoDocument(chords, source=jam_path, jams=jam)


class ChoCoValidHarteChordsCorpus(ChoCoCorpus):
  """
  Iterate over all documents contained in ChoCo to retrieve valid chords in Harte format.
  """
  def __iter__(self) -> Iterator[List[ChoCoDocument]]:
    """
    Yields:
        Iterator[List[HarteAnnotation]]: Progression of chords in Harte format.
    """
    parser = Lark(HARTE_LARK_GRAMMAR, parser='lalr', start="chord")

    for partition_path in self.partitions_paths:
      for jam_path in self._jams_in_partition(partition_path):
        jam = jams.load(jam_path, validate=False)
        namespaces = [ str(a.namespace) for a in jam.annotations ]
        
        chord_namespace = "chord_harte" if "chord_harte" in namespaces else "chord"
        annotation = jam.search(namespace=chord_namespace)
        observations = annotation[0].data
        chords = [HarteAnnotation(obs.value, obs.duration) for obs in observations]
        try:
          [parser.parse(ann.symbol) for ann in chords]
          yield ChoCoDocument(chords, source=jam_path, jams=jam)
        except:
          pass


class ChoCoIsophonicsHarteCorpus(ChoCoCorpus):
  def __iter__(self) -> Iterator[List[ChoCoDocument]]:
    """
    Yields:
        Iterator[List[HarteAnnotation]]: Progression of chords in Harte format.
    """
    isophonics = [p for p in self.partitions_paths if "isophonics" in p]
    for partition_path in isophonics:
      for jam_path in self._jams_in_partition(partition_path):
        jam = jams.load(jam_path, validate=False)
        namespaces = [ str(a.namespace) for a in jam.annotations ]
        
        chord_namespace = "chord_harte" if "chord_harte" in namespaces else "chord"
        annotation = jam.search(namespace=chord_namespace)
        observations = annotation[0].data
        chords = [HarteAnnotation(obs.value, obs.duration) for obs in observations]
        yield ChoCoDocument(chords, source=jam_path, jams=jam)


class ChoCoHarteAnnotationsSectionCorpus(ChoCoCorpus):
  """
  Iterate over all documents contained in ChoCo to retrieve structural annotation of chords.
  Only available in isophonics partition.
  """
  def __iter__(self) -> Iterator[List[ChoCoDocument]]:
    """
    Yields:
        Iterator[List[HarteSectionAnnotation]]: Progression of chords in Harte format.
    """
    structural_partitions = [p for p in self.partitions_paths if "isophonics" in p]
    for partition_path in structural_partitions:
      for jam_path in self._jams_in_partition(partition_path):
        jam = jams.load(jam_path, validate=False)
        namespaces = [ str(a.namespace) for a in jam.annotations ]
        chord_namespace = "chord_harte" if "chord_harte" in namespaces else "chord"
        structure_namespace = "segment_open"
        
        chord_annotations = jam.search(namespace=chord_namespace)[0]
        structure_annotations = jam.search(namespace=structure_namespace)[0]

        chords = list()
        for structure_obs in structure_annotations.data:
          section = structure_obs.value
          start = structure_obs.time
          duration = structure_obs.duration
          end = start + duration

          chords.extend(HarteSectionAnnotation(HarteAnnotation(obs.value, obs.duration), section) 
                        for obs in chord_annotations.slice(start, end))

        yield ChoCoDocument(chords, source=jam_path, jams=jam)


if __name__ == "__main__":
  harte_corpus = ChoCoHarteAnnotationsSectionCorpus("/home/n28div/university/thesis/choco")
  print(next(iter(harte_corpus)))


#def export_corpus(choco_path: str, outfile: str):
#  corpus = ChoCoHarteCorpus(choco_path)
#
#  with open(outfile, "w") as f:
#    for chords in corpus:
#      progression = "|".join(chords) + "\n"
#      f.write(progression)