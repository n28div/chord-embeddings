from typing import List, Tuple
import itertools
import more_itertools as mitertools
import collections
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from harte2vec.harte import HarteToIntervals

def onehot_intervals(intervals):
  if len(intervals) == 0:
    chord_encoding = np.array([0])
  else:
    intervals_encoding = np.stack([np.eye(12)[i, :].sum(axis=0).clip(0, 1) for i in intervals])
    #intervals_encoding = np.stack([np.eye(12)[i, :] for _, i in intervals])
    whole_chord_encoding = intervals_encoding.sum(axis=0).clip(0, 1)
    chord_encoding = np.vstack((intervals_encoding, whole_chord_encoding))
    # convert chord encoding to indexes
    chord_encoding = chord_encoding.dot(2**np.arange(12)[::-1]).astype(int)
  return chord_encoding

class ChocoHarte2VecDataset(Dataset):
  def __init__(self, data: List[str], t: float = 1e-4, c: Tuple[int, int] = (1, 5), k: int = 5):
    super().__init__()
    self._data = data
    self.t = t
    self.c = c
    self.k = k  

    # build vocab and sort to enable faster numpy searching
    counter = dict(sorted(collections.Counter(itertools.chain(*self._data)).items(), key=lambda item: item[0]))

    # turn into numpy array for faster searching
    self.vocab = np.array(list(counter.keys()))
    self.freq = np.array([counter[chord] for chord in self.vocab])
    
    # compute one hot encoding of whole vocabulary
    # use numpy array for faster searching
    self.h2i = HarteToIntervals()
    self.intervals_encoding = np.fromiter((onehot_intervals(self.h2i.convert(chord)) for chord in self.vocab), dtype=np.ndarray)
    self.chord_encoding = np.arange(len(self.vocab))
    
    self._data = self._subsample_data()
    self._data = np.array(self._data) # turn data into numpy array for easier indexing

  def __getitem__(self, index: int):
    data_item = self._data[index]
    # perform negative sampling
    return self._negative_sampling(data_item)

  def __len__(self):
    return len(self._data)

  def _subsample_data(self) -> List[List[str]]:
    # subsample data according to distribution
    prob = 1 - np.sqrt(self.t / self.freq)
    p_w = dict(zip(self.vocab, prob))
    return [[c for c in p if np.random.random() < p_w[c]] for p in self._data]

  def _negative_sampling(self, seq: List[str]):
    if type(self.c) is tuple:
      context_size = 2 * np.random.randint(*self.c) + 1
    else:
      context_size = 2 * self.c + 1
    
    X, y_pos, y_neg = list(), list(), list()

    for chord_context in mitertools.windowed(seq, context_size):
      # extract idx of chords
      chord_context_idxs = np.searchsorted(self.vocab, chord_context)
      
      # extract positives and central chord idx
      chord_idx = chord_context_idxs[context_size // 2]
      positives_index = np.delete(chord_context_idxs, context_size // 2)

      # sample negative samples
      available_negative_indexes = np.delete(np.arange(len(self.vocab)), positives_index)
      negative_p = self.freq[available_negative_indexes] / self.freq[available_negative_indexes].sum()
      negatives_index = np.random.choice(available_negative_indexes, self.k, p=negative_p, replace=False)

      X.append(self.intervals_encoding[chord_idx])
      y_pos.append(self.chord_encoding[positives_index])
      y_neg.append(self.chord_encoding[negatives_index])
    
    return X, y_pos, y_neg

  def collate_fn(self, samples):
    X, y_pos, y_neg = zip(*samples)

    batch_size = len(X)
    context_size = len(X[0])

    # flatten the array and map them to tensors
    X = list(map(torch.tensor, itertools.chain(*X)))
    y_pos = list(map(torch.tensor, itertools.chain(*y_pos)))
    y_neg = list(map(torch.tensor, itertools.chain(*y_neg)))
    # pad each array to have the same lengths for computational efficiency
    # value 0 is 
    X = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=-1)
    y_pos = nn.utils.rnn.pad_sequence(y_pos, batch_first=True, padding_value=-1)
    y_neg = nn.utils.rnn.pad_sequence(y_neg, batch_first=True, padding_value=-1)
    
    return X, y_pos, y_neg


class Harte2VecModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim=300):
    super(Harte2VecModel, self).__init__()
    self.embedding_dim = embedding_dim
    
    self.vocab_size = vocab_size + 1
    self.full_vocab_size = 2**12 + 1
    
    self.embedding_z = nn.EmbeddingBag(self.full_vocab_size, self.embedding_dim, mode="mean")
    self.embedding_v = nn.Embedding(self.vocab_size, self.embedding_dim)
    
  def forward(self, interval, target):
    z = self.embedding_z(interval)
    v = self.embedding_v(target)
    y = torch.einsum("ij,ikj->ik", z, v).reshape(-1)
    return y

class Harte2Vec(object):

  def __init__(self, embedding_dim: int = 300):
    self.embedding_dim = embedding_dim
    
    self._h2i = HarteToIntervals()
    self._vocab = None
    self._chord_embedding = None
    self._intervals_embedding = None

  def train(self, data: List[List[str]], batch_size: int = 5, epochs=1, log_steps=50, subsample_t: float = 1e-4, c: int = (1, 4), k: int = 5):
    dataset = ChocoHarte2VecDataset(data, t=subsample_t, c=c, k=k)
    model = Harte2VecModel(vocab_size=len(dataset.vocab), embedding_dim=self.embedding_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(list(range(epochs)))
    for epoch in progress_bar:
      losses = list()
      for i, batch in enumerate(dataloader):
        progress_bar.set_description_str(f"Step {i:5} / {len(dataset) // batch_size}")
        optimizer.zero_grad()
        
        x, y_pos, y_neg = batch
        # map padding to last embedding
        x[x == -1] = model.full_vocab_size - 1
        y_pos[y_pos == -1] = model.vocab_size - 1
        y_neg[y_neg == -1] = model.vocab_size - 1

        pos_pred = model(x, y_pos)
        pos_loss = loss_fn(pos_pred, torch.ones_like(pos_pred))
        neg_pred = model(x, y_neg)
        neg_loss = loss_fn(neg_pred, torch.zeros_like(neg_pred))
        
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.set_postfix({ "loss": loss.item()})

      mean_loss = sum(losses)/len(losses)
      progress_bar.write(f"Epoch {epoch:5} : Loss {mean_loss:2.5}")
      scheduler.step(mean_loss)

    # training completed
    self._vocab = dataset.vocab
    self._chord_embedding = model.embedding_v
    self._intervals_embedding = model.embedding_z

  def save(self, path: str):
    torch.save({
      "chord_embedding": self._chord_embedding.weight,
      "intervals_embedding": self._intervals_embedding.weight,
      "vocab": self._vocab
    }, path)

  @classmethod
  def from_pretrained(self, path: str):
    h2v = Harte2Vec()
    loaded = torch.load(path)
    h2v._vocab = loaded["vocab"]
    h2v._chord_embedding = nn.Embedding.from_pretrained(loaded["chord_embedding"])
    h2v._intervals_embedding = nn.EmbeddingBag.from_pretrained(loaded["intervals_embedding"], mode="sum")
    return h2v

  def __getitem__(self, chord: str) -> torch.Tensor:
    try:
      idx = int(np.where(self._vocab == chord)[0])
      embedded = self._chord_embedding.weight[idx]
    except:
      # item not available: compute using intervals
      intervals = self._h2i.convert(chord)
      encoded = torch.tensor(onehot_intervals(intervals)).reshape(1, -1)
      embedded = self._intervals_embedding(encoded)
      
    return embedded

  def most_similar(self, chord: str, num: int = 10) -> np.array:
    with torch.no_grad():
      embedding = self[chord]
      dot_prod = np.einsum("i,ji->j", embedding, self._chord_embedding.weight)
      similarity = dot_prod
      similar = np.argsort(similarity)[::-1]
    return self._vocab[similar[:num]], similarity[similar[:num]]

if __name__ == "__main__":
  with open("/tmp/out.txt") as f:
    dataset = [line.replace(" ", "").replace("\n", "").split("|") for line in f.readlines()]
    dataset = [ [chord for chord in seq if len(chord) > 0] for seq in dataset if len(seq) > 15 ]

  h2v = Harte2Vec()
  h2v.train(dataset, batch_size=256, epochs=1, c=6, k=20)
  h2v.save("/tmp/out")
  #print(h2v.most_similar("C:maj"))
  #del h2v

  #h2v = Harte2Vec.from_pretrained("/tmp/out")
  #print(h2v.most_similar("C:maj"))
  #print(h2v["C:maj"][0])
  #print(h2v["C:maj7(13)"])
  #pass
