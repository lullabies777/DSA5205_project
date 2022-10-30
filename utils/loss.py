import torch 
import numpy as np 

def max_sharpe(y_return, weights, annual_rate = 0.04):
    weights = torch.unsqueeze(weights, 1)
    meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)
    covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return]).to("cuda")
    portReturn = torch.matmul(weights, meanReturn)
    portVol = torch.matmul(
        weights, torch.matmul(covmat, torch.transpose(weights, 2, 1))
    )
    rf = (1 + annual_rate) ** (1 / 365) - 1
    objective = (portReturn - rf) / (torch.sqrt(portVol))
    return -objective.mean()