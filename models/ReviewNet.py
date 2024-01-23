import torch.nn as nn

class ReviewNet(nn.Module):
  def __init__(self, input_size=768, output_size=2, hidden_size=256):
    super().__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.output_size = output_size
    self.metric_scores = None
    self.base_net = nn.Sequential(
        nn.Linear(self.input_size, self.hidden_size),
        nn.BatchNorm1d(self.hidden_size),
        nn.Dropout(),
        nn.ReLU(),

        nn.Linear(self.hidden_size, self.hidden_size),
        nn.BatchNorm1d(self.hidden_size),
        nn.Dropout(),
        nn.ReLU(),

        nn.Linear(self.hidden_size, self.hidden_size),
        nn.BatchNorm1d(self.hidden_size),
        nn.Dropout(),
        nn.ReLU(),

        nn.Linear(self.hidden_size, 6),
        nn.Softmax(dim=1)
    )

  def forward(self, X):
    res = self.base_net(X)
    return res