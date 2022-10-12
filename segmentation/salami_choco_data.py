import pandas as pd
import os.path as osp
import os
import jams
import csv
import itertools
from glob import glob
import pickle
from functools import partial
import copy
import collections

CHOCO_PATH = "../choco/"
SALAMI_PATH = "../salami-data-public/"

Chord = collections.namedtuple("Chord", ["symbol", "duration"])
AnnotatedChord = collections.namedtuple("AnnotatedChord", ["chord", "section"])

class Dataset(object):
  def __init__(self, salami_path: str, choco_path: str):
    # TODO: Check dir exists
    self.salami_path = salami_path
    self.choco_path = choco_path
    
    isophonics_salami_meta = pd.read_csv(osp.join(self.salami_path, "metadata/id_index_isophonics.csv"))
    isophonics_choco_meta = pd.read_csv(osp.join(self.choco_path, "partitions/isophonics/choco/meta.csv"))
    self.meta = isophonics_choco_meta.join(isophonics_salami_meta.set_index("TITLE"), on="file_title").dropna()

    self._load_data()

  def _load_salami(self, path: str):
    with open(path) as f:
      reader = csv.reader(f, delimiter='\t')
      salami = [
        (float(start_t), float(end_t), start_l)
        for (start_t, start_l), (end_t, _) in itertools.pairwise(reader)
      ]
    return salami

  def _match_sections_with_chords(self, jam, salami):
    sections = list()
    for section in salami:
      start, end, name = section
      if start < end:
        for chord_data in jam.slice(start, end).data:
          chord = Chord(chord_data.value, chord_data.duration)
          ann = AnnotatedChord(chord, name)
          sections.append(ann)
      
    return sections

  def _load_data(self):
    self.data = list()
    
    for idx, row in self.meta.iterrows():
      jams_rel_path = "/".join(row.jams_path.split("/")[2:])
      jams_path = osp.join(self.choco_path, jams_rel_path)
      
      jam = jams.load(jams_path, validate=False)
      chord_ns = jam.search(namespace="chord")[0]
      epsilon = min([obs.duration for obs in chord_ns.data]) / 100

      piece = dict()
      piece["title"] = row.file_title
      piece["artist"] = row.file_performer
      piece["jams"] = jam
    
      salami_textfiles = glob(osp.join(self.salami_path, "annotations", str(int(row.SONG_ID)), "parsed", "textfile*_functions.txt"))
      for salami_textfile in salami_textfiles:
        salami = self._load_salami(salami_textfile)
        piece["salami"] = salami
        self.data.append(piece)

  def iter(self, filter_chord = None):
    for sample in self.data:
      jam = copy.deepcopy(sample["jams"])
      chord_ns = jam.search(namespace="chord")[0]

      durations = [obs.duration for obs in chord_ns.data]
      epsilon = min(durations) / 100

      if filter_chord is not None:
        obs = chord_ns.pop_data()
        obs = list(filter(partial(filter_chord, obs), obs))
        
        # fix skewed durations
        if obs[0].time != 0:
          obs[0] = jams.Observation(
            time=0,
            duration=obs[0].time + obs[0].duration,
            value=obs[0].value,
            confidence=obs[0].confidence
          )

        for i in range(1, len(obs)):
          should_start = obs[i - 1].time + obs[i - 1].duration
          if obs[i].time > should_start:
            additional_time = obs[i].time - should_start
            obs[i] = jams.Observation(
              time=should_start,
              duration=obs[i].duration + additional_time,
              value=obs[i].value,
              confidence=obs[i].confidence
            )
        
        chord_ns.append_records(obs)

      yield {
        "title": sample["title"],
        "artist": sample["artist"],
        "structure": self._match_sections_with_chords(chord_ns, sample["salami"])
      }

  def __len__(self):
    return len(self.data)

if __name__ == "__main__":
  def filter_out_short_chords(chords, obs):
    return obs.duration < 4

  data = Dataset(SALAMI_PATH, CHOCO_PATH)
  original = next(data.iter())
  short = next(data.iter(filter_chord=filter_out_short_chords))
  
  assert sum(map(lambda s: len(s["chords"]), data.iter(filter_chord=filter_out_short_chords))) < sum(map(lambda s: len(s["chords"]), data.iter()))
  
  print(original, short)
