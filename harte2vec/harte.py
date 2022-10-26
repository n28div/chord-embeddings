from typing import List, Tuple
from lark import Lark, Transformer
import numpy as np
import more_itertools as mitertools
import itertools
import re

HARTE_LARK_GRAMMAR = """
chord: note ":" SHORTHAND ("(" degree_list ")")? ("/" bass)?
     | note ":" "(" degree_list ")" ("/" bass)?
     | note ("/" bass)?
     | NA
note: NATURAL | NATURAL MODIFIER
NATURAL: "A" | "B" | "C" | "D" | "E" | "F" | "G"
MODIFIER: "b" | "#"
NA: "N" | "X"
bass: degree
degree_list: degree ("," degree)*
degree: MISSING? MODIFIER* INTERVAL
MISSING: "*"
INTERVAL: "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "13"
SHORTHAND: "maj" | "min" | "dim" | "aug" | "maj7" | "min7" | "7" | "dim7" | "hdim7"
         | "minmaj7" | "maj6" | "min6" | "9" | "maj9" | "min9" | "sus4"
         | "hdim" |"sus2" | "min11" | "maj11" | "11" | "min13" | "maj13" | "13"
         | "5" | "1"
%ignore " "
"""

class TreeToHarteChord(Transformer):
  """
  Lark transformer to turn a parse tree into an Harte chord representation.
  The representation consists of a dict with keys:
    * root
        Root note of the chord
    * shorthand - OPTIONAL
        Shorthand of the chord
    * bass - OPTIONAL
        Modified bass note if slash chord is used
    * degrees - OPTIONAL
        Modified degrees on the chord (with missing degrees identified with * i.e. *3
  """
  NATURAL = str
  MODIFIER = str
  MISSING = str
  SHORTHAND = lambda self, sh: { "shorthand": str(sh) }
  INTERVAL = str
  degree = lambda self, elems: "".join(elems)
  bass = lambda self, elems: {"bass": "".join(elems)}
  note = lambda self, elems: {"root": "".join(elems)}
  degree_list = lambda self, elems: elems
  
  def chord(self, elems):
    d = dict()
    for elem in elems:
      if isinstance(elem, dict):
        d.update(**elem)
      elif isinstance(elem, list):
        d.update({"degrees": list(mitertools.collapse(elem))})
  
    return d


SHORTHAND_COMPONENTS = {
  "maj": ["3", "5"],
  "min": ["b3", "5"],
  "dim": ["b3", "b5"],
  "aug": ["3", "#5"],
  "maj7": ["3", "5", "7"],
  "min7": ["b3", "5", "b7"],
  "7": ["3", "5", "b7"],
  "dim7": ["b3", "b5", "bb7"],
  "hdim7": ["b3", "b5", "b7"],
  "minmaj7": ["b3", "5", "7"],
  "maj6": ["3", "5", "6"],
  "min6": ["b3", "5", "6"],
  "9": ["3", "5", "b7", "9"],
  "maj9": ["3", "5", "7", "9"],
  "min9": ["b3", "5", "b7", "9"],
  "sus4": ["4", "5"],
  # new shorthands based on ChoCo data
  "sus2": ["2", "5"],
  "hdim": ["b3", "b5", "b7"],
  "11": ["3", "5", "b7", "9", "11"],
  "maj11": ["3", "5", "7", "9", "11"],
  "min11": ["b3", "5", "b7", "9", "11"],
  "13": ["3", "5", "b7", "9", "11"],
  "maj13": ["3", "5", "7", "9", "11", "13"],
  "min13": ["b3", "5", "b7", "9", "11", "13"],
  "5": ["5"],
  "1": []
}

# idx is the pitch of the note, where 0 is C ans 10 is B
class NoteToSemitones(dict):
  def __init__(self):
    self.NOTE_RE = re.compile(r"(A|B|C|D|E|F|G)")
    self.FLAT_RE = re.compile(r"(b)")
    self.SHARP_RE = re.compile(r"(#)")
    self._note_map = {
      "C": 0,
      "D": 2,
      "E": 4,
      "F": 5,
      "G": 7,
      "A": 9,
      "B": 10
    }

  def get(self, note):
    # extract numeric interval
    numeric_note = self.NOTE_RE.search(note).group()
    numeric_note = self._note_map[numeric_note]

    for _ in self.SHARP_RE.findall(note): numeric_note = (numeric_note + 1) % 12
    for _ in self.FLAT_RE.findall(note): numeric_note = (numeric_note - 1) % 12
    
    return numeric_note

NOTE2IDX = NoteToSemitones()



# interval to semitones maps the number of semitones to move when an interval
# is specified
class IntervalToSemitones(dict):
  def __init__(self):
    self.INTERVAL_RE = re.compile(r"(\d+)")
    self.FLAT_RE = re.compile(r"(b)")
    self.SHARP_RE = re.compile(r"(#)")
    self._interval_map = {
      "1": 0,
      "2": 2,
      "3": 4,
      "4": 5,
      "5": 7,
      "6": 9,
      "7": 11,
      "8": 12,
      "9": 14,
      "10": 16,
      "11": 17,
      "13": 21
    }

  def get(self, interval):
    # extract numeric interval
    numeric_interval = self.INTERVAL_RE.search(interval).group()
    numeric_interval = self._interval_map[numeric_interval]

    for _ in self.SHARP_RE.findall(interval): numeric_interval += 1
    for _ in self.FLAT_RE.findall(interval): numeric_interval -= 1
    
    return numeric_interval

INTERVAL2SEMITONES = IntervalToSemitones()

class HarteToIntervals(object):
  """
  Convert a chord from Harte embedding to a set of constituents of the chord
  in the form (root, note) where root and node are two integers each encoded 
  as the number of semitones from C. 
  Chords are hence represented as the set of intervals from the root that they're 
  composed of.
  """
  def __init__(self):
    self.harte_parser = Lark(HARTE_LARK_GRAMMAR, 
                             parser='lalr',
                             start="chord",
                             propagate_positions=False,
                             maybe_placeholders=False,
                             transformer=TreeToHarteChord())
    self._already_converted = dict()

  def convert(self, chord: str) -> List[Tuple[str, str]]:
    """
    Convert a chord to a list of intervals. 
    Intervals and notes are encoded as  the semitones of distance from C.

    Args:
        chord (str): Input chord in Harte form
    Raises:
        ValueError: Chord is not recognized as a valid Harte chord
    Returns:
        List[Tuple[str, str]]: List of intervals that make up the chord
    """
    if chord in self._already_converted: return self._already_converted[chord]

    try:
      parsed = self.harte_parser.parse(chord)
    except:
      raise ValueError(f"{chord} is not a valid Harte chord.")

    converted = []
    if len(parsed) > 0:
      root = parsed["root"]
      root_idx = NOTE2IDX.get(root)

      components = mitertools.collapse([
        "1", # for root
        SHORTHAND_COMPONENTS[parsed.get("shorthand", "maj")],
        parsed.get("degrees", []),
        parsed.get("bass", [])])

      # filter out removed components
      components = list(filter(lambda c: "*" not in c, components))
      
      # map components to semitones
      semitones = list(map(INTERVAL2SEMITONES.get, components))
      converted = [(root_idx, (root_idx + s) % 12) for s in semitones ]
    
    return converted
    


class HarteDecomposer(object): 
  def __init__(self):
    self._embeddings = dict()
    self.harte_parser = Lark(HARTE_LARK_GRAMMAR, 
                             parser='lalr',
                             start="chord",
                             propagate_positions=False,
                             maybe_placeholders=False,
                             transformer=TreeToHarteChord())
    
  @property
  def note_idx(self):
    return {
      "C": 0,
      "C#": 1, "Db": 1,
      "D": 2,
      "D#": 3, "Eb": 3,
      "E": 4,
      "F": 5,
      "F#": 6, "Gb": 6,
      "G": 7,
      "G#": 8, "Ab": 8,
      "A": 9,
      "A#": 10, "Bb": 10,
      "B": 10
    }
  
  @property
  def shorthand_components(self):
    return {
      "": [],
      "maj": ["3", "5"],
      "min": ["b3", "5"],
      "dim": ["b3", "b5"],
      "aug": ["3", "#5"],
      "maj7": ["3", "5", "7"],
      "min7": ["b3", "5", "b7"],
      "7": ["3", "5", "b7"],
      "dim7": ["b3", "b5", "bb7"],
      "hdim7": ["b3", "b5", "b7"],
      "minmaj7": ["b3", "5", "7"],
      "maj6": ["3", "5", "6"],
      "min6": ["b3", "5", "6"],
      "9": ["3", "5", "b7", "9"],
      "maj9": ["3", "5", "7", "9"],
      "min9": ["b3", "5", "b7", "9"],
      "sus4": ["4", "5"]
    }
  
  @property
  def interval_to_semitones(self):
    return {
      "1": 0, "bb2": 0,
      "b2": 1,
      "2": 2, "bb3": 2,
      "b3": 3, "#2": 3,
      "3": 4, "b4": 4,
      "4": 5, "#3": 5,
      "b5": 6, "#4": 6,
      "5": 7, "bb6": 7,
      "b6": 8, "#5": 8,
      "6": 9, "bb7": 9,
      "b7": 10, "#6": 19,
      "7": 11,
      "bb9": 12,
      "b9": 13,
      "9": 14, "bb10": 14,
      "b10": 15, "#9": 15,
      "10": 16, "b11": 16,
      "11": 17, "#10": 17,
      "#11": 18,
      "b13": 20,
      "13": 21
    }
     
  def __getitem__(self, key):
    try:
      parsed = self.harte_parser.parse(key)
    except:
      raise ValueError(f"{key} is not a valid Harte chord.")
    
    embedding = np.zeros(11)

    # if it's not an empty chord
    if len(parsed) > 0:
      root_idx = self.note_idx[parsed["root"]] 
      embedding[root_idx] = 1

      degrees = parsed.get("degrees", [])
      bass = [parsed.get("bass")] if "bass" in parsed else []
      shorthand = parsed.get("shorthand", "maj" if len(degrees) == 0 else "")
      shorthand = self.shorthand_components[shorthand]

      components = set(shorthand).union(degrees).union(bass)
      for component in components:
        shift = self.interval_to_semitones[component.replace("*", "")]
        embedding[(root_idx + shift) % 11] += -1 if "*" in component else 1
      
      embedding = np.clip(embedding, 0, 1)

    return embedding
      
#emb = HarteEmbedding()
#assert (emb["C:min7"] == emb["C:(b3,5,b7)"]).all()
#assert (emb["C:min7(*5,11)"] == emb["C:(b3,b7,11)"]).all()
#assert (emb["C"] == emb["C:maj"]).all()
#assert (emb["C:maj"] == emb["C:(3, 5)"]).all()
#assert (emb["A/3"] == emb["A:maj/3"]).all()
#assert (emb["A:(3,5)/3"] == emb["A:maj/3"]).all()
#assert (emb["C:maj(4)"] == emb["C:(3,4,5)"]).all()
#assert (emb["N"] == np.zeros(11)).all()

if __name__ == "__main__":
  hti = HarteToIntervals()
  print(hti.convert("G:maj"))