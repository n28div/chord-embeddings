import numpy as np
from gensim.models import KeyedVectors
from pitchclass2vec.model.base import BaseModel
from pitchclass2vec.data import ChocoChordDataset


class Pitchclass2VecModel(KeyedVectors):
  def __init__(self, encoding_model: ChocoChordDataset, embedding_model: BaseModel, model_path: str):
    """
    Initialize a pitchclass2vec model as a gensim KeyedVectors.
    Key informations are extracted from the trained checkpoint
    from the specified model_path.

    Args:
        model_path (str): Path to the model to be loaded.
    """
    self.model_path = model_path
    self.encoding_model = encoding_model
    self.embedding_model = embedding_model.load_from_checkpoint(model_path)

    super().__init__(self.embedding_model.embedding_dim)

  def has_index_for(self, key: str) -> bool:
    """
    Method to check if the model has an element in its vocabulary.
    Always true for pitchclass2vec.

    Args:
        key (str): Chord that is being checked.

    Returns: True
    """
    return True

  def get_vector(self, key: str, norm: bool = False) -> np.array:
    """
    Get the vector corresponding to the provided chord.

    Args:
        key (str): Provided chord.
        norm (bool, optional): Wether output vector should be normalized. Defaults to False.

    Returns:
        np.array: Output vector.
    """
    encoded_chord = self.encoding_model.encode_chord(key)
    vector = self.embedding_model[encoded_chord]
    if norm:
      result = vector / np.linalg.norm(vector)
    
    return vector
