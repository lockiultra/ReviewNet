import numpy as np
from torch.utils.data import Dataset

class ReviewsDataset(Dataset):
  def __init__(self, X, y):
    self.X = np.array(X)
    self.y = np.array(y)

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return (self.X[idx], self.y[idx])