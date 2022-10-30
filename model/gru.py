import torch  
import torch.nn as nn
import torch.nn.functional as F 

class GRU(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.n_layers = config.n_layers
    self.hidden_dim = config.hidden_dim
    self.lb = config.lb 
    self.ub = config.ub 
    self.gru = nn.GRU(
        config.n_stocks, config.hidden_dim, num_layers = config.n_layers, batch_first = True, bidirectional = config.bidirectional
    )
    self.dropout = nn.Dropout(config.dropout)
    self.scale = 2 if config.bidirectional else 1 
    self.fc = nn.Linear(config.hidden_dim * self.scale, config.n_stocks)
    self.swish = nn.SiLU() 
  
  def forward(self, x):
      init_h = torch.zeros(self.n_layers * self.scale, x.size(0), self.hidden_dim, device = x.device)
      x, _ = self.gru(x, init_h)
      h_t = x[:, -1, :]
      logit = self.fc(self.dropout(h_t))
      logit = self.swish(logit)
      logit = F.softmax(logit, dim=-1)
      logit = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in logit])
      return logit
    
  def rebalance(self, weight, lb, ub):
    old = weight
    weight_clamped = torch.clamp(old, lb, ub)
    while True:
        leftover = (old - weight_clamped).sum().item()
        nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
        gift = leftover * (nominees / nominees.sum())
        weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
        old = weight_clamped
        if len(torch.where(weight_clamped > ub)[0]) == 0:
            break
        else:
            weight_clamped = torch.clamp(old, lb, ub)
    return weight_clamped
  